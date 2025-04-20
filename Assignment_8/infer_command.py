import os
import string
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import mlflow
import mlflow.tensorflow
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, img_width=128, img_height=32):
        self.img_width = img_width
        self.img_height = img_height
        
        self.char_to_num = layers.StringLookup(
            vocabulary=list(string.ascii_lowercase + string.digits + " "), mask_token=None
        )
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )


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


def process_image(image_path, config):
    """Process image for model input."""
    try:
        img = Image.open(image_path)
        img = img.convert('L')
        img_array = np.array(img)
        
        # add channel dimension if needed
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)
        
        # convert to tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_tensor = img_tensor / 255.0  # mormalize to [0, 1]
        
        # resize image
        img_tensor = distortion_free_resize(img_tensor, (config.img_width, config.img_height))
        
        # add batch dimension
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        
        return img_tensor
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


def decode_predictions(pred, config):
    """Decode model predictions to text."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # use greedy search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    
    # convert to text
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.not_equal(res, -1)))
        res = tf.squeeze(res)
        res = tf.strings.reduce_join(config.num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
        
    return output_text[0] if output_text else ""


def load_model(model_uri):
    """Load model from MLflow model registry."""
    try:
        logger.info(f"Loading model from {model_uri}")
        model = mlflow.tensorflow.load_model(model_uri)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def main():
    """Main function to run inference on a single image."""
    parser = argparse.ArgumentParser(description="Handwriting recognition inference")
    parser.add_argument("--model_uri", type=str, required=True, 
                        help="MLflow model URI (e.g., 'runs:/run_id/handwriting_recognition_model' or 'models:/HandwritingRecognitionModel/1')")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--img_width", type=int, default=128, help="Image width")
    parser.add_argument("--img_height", type=int, default=32, help="Image height")
    
    args = parser.parse_args()
    
    try:
        # check if image exists
        if not os.path.exists(args.image_path):
            logger.error(f"Image not found: {args.image_path}")
            return
        
        config = Config(img_width=args.img_width, img_height=args.img_height)
        
        model = load_model(args.model_uri)
        
        logger.info(f"Processing image: {args.image_path}")
        processed_img = process_image(args.image_path, config)
        
        predictions = model.predict(processed_img)
        
        # decode prediction
        recognized_text = decode_predictions(predictions, config)
        
        logger.info(f"Recognized text: {recognized_text}")
        print(f"\nRecognized text: {recognized_text}")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")


if __name__ == "__main__":
    main()