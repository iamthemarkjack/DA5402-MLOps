import os
import string
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import argparse
import logging
from datetime import datetime
import uuid
import zipfile
import tarfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self,
                 batch_size=16,
                 img_width=128,
                 img_height=32,
                 max_length=21,
                 padding_token=99,
                 downsample_factor=4,
                 train_ratio=0.8,
                 val_ratio=0.1,
                 epochs=10,
                 early_stopping_patience=2,
                 reduce_lr_patience=2,
                 experiment_name="handwriting_recognition"):
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.max_length = max_length
        self.padding_token = padding_token
        self.downsample_factor = downsample_factor
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.experiment_name = experiment_name

        self.char_to_num = layers.StringLookup(
            vocabulary=list(string.ascii_lowercase + string.digits + " "), mask_token=None        
        )
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        self.characters = len(self.char_to_num.get_vocabulary())

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def prepare_dataset(config, images, labels, split_idx=None):
    """Prepare dataset for train validation and test"""

    def encode_single_image(img_path, label):
        # read and process the image
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = distortion_free_resize(img, (config.img_width, config.img_height))

        # process the label
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        label = config.char_to_num(label)
        length = keras.ops.shape(label)[0]
        pad_amount = config.max_length - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=config.padding_token)

        return {"image": img, "label": label}
        

    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).map(encode_single_image, 
                                                            num_parallel_calls=tf.data.AUTOTUNE)

    if split_idx is not None:
        dataset = dataset.take(split_idx)

    return dataset.batch(config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = keras.ops.cast(keras.ops.shape(y_true)[0], dtype="int64")
        input_length = keras.ops.cast(keras.ops.shape(y_pred)[1], dtype="int64")
        label_length = keras.ops.cast(keras.ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * keras.ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * keras.ops.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

def build_model(config):
    """Build the handwriting recognition model"""
    # inputs
    input_img = layers.Input(shape=(config.img_width, config.img_height, 1), name="image")
    labels = layers.Input(name="label", shape=(None,))
    
    # CNN
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    
    # RNN
    new_shape = ((config.img_width // 4), (config.img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25), name="lstm1")(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25), name="lstm2")(x)
    
    # output
    x = layers.Dense(config.characters + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)
    
    # model for training
    model = keras.Model(
        inputs=[input_img, labels],
        outputs=output,
        name="handwriting_recognizer"
    )
    
    model.compile(optimizer=keras.optimizers.Adam())
    
    # model for inference
    prediction_model = keras.Model(
        inputs=input_img,
        outputs=x,
        name="prediction_model"
    )
    
    return model, prediction_model


def decode_batch_predictions(pred, config):
    """Decode predicted sequences"""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # greedy search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.not_equal(res, -1)))
        res = tf.squeeze(res)
        res = tf.strings.reduce_join(config.num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def calculate_edit_distance(y_true, y_pred):
    """Calculate edit distance between true and predicted texts"""
    edit_distances = []
    for true_text, pred_text in zip(y_true, y_pred):
        edit_distances.append(tf.edit_distance(
            tf.sparse.from_dense(tf.strings.unicode_split(pred_text, 'UTF-8')),
            tf.sparse.from_dense(tf.strings.unicode_split(true_text, 'UTF-8'))
        ).numpy())
    return np.mean(edit_distances)


def load_data():
    """Load IAM dataset"""
    zip_path = keras.utils.get_file(
        "IAM_Words.zip",
        "https://github.com/sayakpaul/Handwriting-Recognizer-in-Keras/releases/download/v1.0.0/IAM_Words.zip"
    )
    extract_dir = os.path.join(zip_path, "iam_data")
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    data_dir = os.path.join(extract_dir, "IAM_Words")
    words_dir = os.path.join(data_dir, "words")
    if not os.path.exists(words_dir):
        with tarfile.open(data_dir + "/words.tgz", "r:gz") as tar:
            tar.extractall(words_dir)
    
    # load words.txt file
    words_file = os.path.join(data_dir, "words.txt")
    words = []
    with open(words_file, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            parts = line.strip().split(" ")
            if len(parts) >= 9 and parts[1] != "err":
                path_parts = parts[0].split("-")
                word_rel_path = os.path.join(path_parts[0], path_parts[0] + '-' + path_parts[1], parts[0])
                word_text = parts[8].lower()
                image_path = os.path.join(data_dir, "words", word_rel_path + ".png")
                if os.path.exists(image_path):
                    words.append((image_path, word_text))
    
    images = np.array([item[0] for item in words])
    labels = np.array([item[1] for item in words])
    
    return images, labels


def split_data(images, labels, train_ratio=0.8, val_ratio=0.1):
    """Split data into train, validation and test sets"""
    # shuffle data
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]
    
    # split data
    num_samples = len(images)
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)
    
    train_images = images[:num_train]
    train_labels = labels[:num_train]
    
    val_images = images[num_train:num_train + num_val]
    val_labels = labels[num_train:num_train + num_val]
    
    test_images = images[num_train + num_val:]
    test_labels = labels[num_train + num_val:]
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


class CTCCallback(keras.callbacks.Callback):
    """Custom callback to compute CER during training"""
    
    def __init__(self, prediction_model, validation_dataset, config):
        super().__init__()
        self.prediction_model = prediction_model
        self.validation_dataset = validation_dataset
        self.config = config
        self.edit_distances = []
    
    def on_epoch_end(self, epoch, logs=None):
        predictions = []
        targets = []
        
        for batch in self.validation_dataset:
            batch_images = batch["image"]
            batch_labels = batch["label"]
            
            # get predictions
            preds = self.prediction_model.predict(batch_images)
            pred_texts = decode_batch_predictions(preds, self.config)
            predictions.extend(pred_texts)
            
            # get true labels
            for label in batch_labels:
                label = tf.gather(label, tf.where(tf.not_equal(label, 0)))
                label = tf.squeeze(label)
                label = tf.strings.reduce_join(self.config.num_to_char(label)).numpy().decode("utf-8")
                targets.append(label)
        
        # calculate edit distance
        edit_distance = calculate_edit_distance(targets, predictions)
        self.edit_distances.append(edit_distance)
        
        print(f"Mean edit distance: {edit_distance:.4f}")
        # log using MLflow
        mlflow.log_metric("edit_distance", edit_distance, step=epoch)


def train_model(config, experiment_id=None):
    """Train handwriting recognition model with MLflow tracking"""
    try:
        # set up MLflow
        if experiment_id:
            mlflow.set_experiment_tag("experiment_id", experiment_id)
        else:
            mlflow.set_experiment(config.experiment_name)
        
        run_id = str(uuid.uuid4())[:8]
        run_name = f"handwriting_run_{run_id}"
        
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"Starting MLflow run: {run_name}")
            
            # log parameters
            mlflow.log_param("batch_size", config.batch_size)
            mlflow.log_param("img_width", config.img_width)
            mlflow.log_param("img_height", config.img_height)
            mlflow.log_param("max_length", config.max_length)
            mlflow.log_param("downsample_factor", config.downsample_factor)
            mlflow.log_param("train_ratio", config.train_ratio)
            mlflow.log_param("val_ratio", config.val_ratio)
            mlflow.log_param("test_ratio", config.test_ratio)
            mlflow.log_param("epochs", config.epochs)
            
            # load and split data
            logger.info("Loading dataset...")
            images, labels = load_data()
            logger.info(f"Dataset loaded: {len(images)} samples")
            
            (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = split_data(
                images, labels, train_ratio=config.train_ratio, val_ratio=config.val_ratio
            )
            
            logger.info(f"Train samples: {len(train_images)}")
            logger.info(f"Validation samples: {len(val_images)}")
            logger.info(f"Test samples: {len(test_images)}")
            
            # log dataset split sizes
            mlflow.log_param("num_train_samples", len(train_images))
            mlflow.log_param("num_val_samples", len(val_images))
            mlflow.log_param("num_test_samples", len(test_images))
            
            # prepare datasets
            logger.info("Preparing datasets...")
            train_dataset = prepare_dataset(config, train_images, train_labels)
            validation_dataset = prepare_dataset(config, val_images, val_labels)
            test_dataset = prepare_dataset(config, test_images, test_labels)
            
            # build model
            logger.info("Building model...")
            model, prediction_model = build_model(config)
            
            # callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=config.early_stopping_patience, 
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    patience=config.reduce_lr_patience,
                    factor=0.2,
                    verbose=1
                ),
                CTCCallback(prediction_model, validation_dataset, config),
                keras.callbacks.TensorBoard(log_dir=f"./logs/{run_name}"),
                keras.callbacks.ModelCheckpoint(
                    filepath=f"./checkpoints/{run_name}/model_{{epoch:02d}}_{{val_loss:.4f}}.h5",
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ),
                MLflowCallback(log_dir=f"./logs/{run_name}")
            ]
            
            # train model
            logger.info("Training model...")
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=config.epochs,
                callbacks=callbacks
            )
            
            # log history
            for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            # plot and log figures
            # loss plot
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            loss_plot_path = f"loss_plot_{run_id}.png"
            plt.savefig(loss_plot_path)
            mlflow.log_artifact(loss_plot_path)
            
            # edit distance plot
            plt.figure(figsize=(12, 6))
            edit_distances = callbacks[2].edit_distances
            plt.plot(range(len(edit_distances)), edit_distances)
            plt.title('Average Edit Distance')
            plt.xlabel('Epoch')
            plt.ylabel('Edit Distance')
            edit_distance_plot_path = f"edit_distance_plot_{run_id}.png"
            plt.savefig(edit_distance_plot_path)
            mlflow.log_artifact(edit_distance_plot_path)
            
            # evaluate on test set
            logger.info("Evaluating model on test set...")
            test_loss = model.evaluate(test_dataset)
            mlflow.log_metric("test_loss", test_loss)
            
            # compute edit distance on test set
            test_predictions = []
            test_targets = []
            
            for batch in test_dataset:
                batch_images = batch["image"]
                batch_labels = batch["label"]
                
                preds = prediction_model.predict(batch_images)
                pred_texts = decode_batch_predictions(preds, config)
                test_predictions.extend(pred_texts)
                
                for label in batch_labels:
                    label = tf.gather(label, tf.where(tf.not_equal(label, 0)))
                    label = tf.squeeze(label)
                    label = tf.strings.reduce_join(config.num_to_char(label)).numpy().decode("utf-8")
                    test_targets.append(label)
            
            test_edit_distance = calculate_edit_distance(test_targets, test_predictions)
            mlflow.log_metric("test_edit_distance", test_edit_distance)
            logger.info(f"Test edit distance: {test_edit_distance:.4f}")
            
            # log model
            logger.info("Logging model to MLflow...")
            mlflow.tensorflow.log_model(
                prediction_model,
                "handwriting_recognition_model",
                registered_model_name="HandwritingRecognitionModel"
            )
            
            # save sample test images for inference demo
            os.makedirs("test_samples", exist_ok=True)
            for i in range(min(5, len(test_images))):
                tf.io.write_file(f"test_samples/sample_{i}.png", tf.io.read_file(test_images[i]))
                with open(f"test_samples/sample_{i}_text.txt", "w") as f:
                    f.write(test_labels[i])
            
            # log sample test images
            mlflow.log_artifacts("test_samples", "test_samples")
            
            logger.info(f"Training completed successfully. Run ID: {run.info.run_id}")
            
            return run.info.run_id
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


class MLflowCallback(keras.callbacks.Callback):
    """Custom MLflow callback to log artifacts during training"""
    
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
    
    def on_epoch_end(self, epoch, logs=None):
        # log TensorBoard data
        if epoch % 10 == 0 and epoch > 0:
            mlflow.log_artifacts(self.log_dir, f"tensorboard_logs_epoch_{epoch}")


def main():
    """Main function to run the training"""
    parser = argparse.ArgumentParser(description="Train handwriting recognition model with MLflow tracking")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--img_width", type=int, default=128, help="Image width")
    parser.add_argument("--img_height", type=int, default=32, help="Image height")
    parser.add_argument("--max_length", type=int, default=21, help="Maximum length of sequences")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data ratio")
    parser.add_argument("--experiment_name", type=str, default="handwriting_recognition", help="MLflow experiment name")
    parser.add_argument("--experiment_id", type=str, default=None, help="MLflow experiment ID")
    
    args = parser.parse_args()
    
    # create config from args
    config = Config(
        batch_size=args.batch_size,
        img_width=args.img_width,
        img_height=args.img_height,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        experiment_name=args.experiment_name
    )
    
    # train model
    train_model(config, args.experiment_id)

if __name__ == "__main__":
    main()