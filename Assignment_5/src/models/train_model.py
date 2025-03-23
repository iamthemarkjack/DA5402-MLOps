import os
import argparse
import logging
import json
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/train_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("train_model")

def load_params():
    """
    Load parameters from params.yaml
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def load_data():
    """
    Load training and validation data
    """
    train_df = pd.read_pickle("data/train/train.pkl")
    val_df = pd.read_pickle("data/val/val.pkl")
    
    X_train = np.stack(train_df['image_data'].values)
    y_train = train_df['class_idx'].values
    
    X_val = np.stack(val_df['image_data'].values)
    y_val = val_df['class_idx'].values
    
    # normalize the data
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    
    # convert labels to one-hot encoding
    num_classes = len(np.unique(y_train))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    
    logger.info(f"Loaded data: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
    logger.info(f"Validation data: X_val.shape={X_val.shape}, y_val.shape={y_val.shape}")
    
    return X_train, y_train, X_val, y_val, num_classes

def create_model(params, num_classes):
    """
    Create a CNN model based on parameters
    """
    model = models.Sequential()
    
    # input layer
    model.add(layers.Conv2D(params['conv_filters'][0], (params['kernel_sizes'][0], params['kernel_sizes'][0]), 
                           padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(params['dropout_rate']))
    
    # add additional convolutional layers
    for i in range(1, params['conv_layers']):
        if i < len(params['conv_filters']) and i < len(params['kernel_sizes']):
            model.add(layers.Conv2D(params['conv_filters'][i], 
                                  (params['kernel_sizes'][i], params['kernel_sizes'][i]), 
                                  padding='same', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Dropout(params['dropout_rate']))
    
    # flatten and add dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_single_model(model_params, X_train, y_train, X_val, y_val, seed, run_name):
    """
    Train a single model with given parameters
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    model = create_model(model_params, y_train.shape[1])
    
    # create directories for model artifacts
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs(f"models/{run_name}", exist_ok=True)
    
    # callbacks
    checkpoint = ModelCheckpoint(
        f"models/checkpoints/{run_name}_best.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=model_params['epochs'],
        batch_size=model_params['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    model.save(f"models/{run_name}/model.keras")
    
    # save training history
    with open(f"models/{run_name}/history.json", 'w') as f:
        history_dict = {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        json.dump(history_dict, f)
    
    # plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"models/{run_name}/training_history.png")
    
    # evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    logger.info(f"Run '{run_name}' completed with validation accuracy: {val_accuracy:.4f}")
    
    return {
        'model_path': f"models/{run_name}/model.keras",
        'val_accuracy': float(val_accuracy),
        'val_loss': float(val_loss),
        'params': model_params
    }

def hyperparameter_tuning(model_params, X_train, y_train, X_val, y_val, tuning_params, seed):
    """
    Perform hyperparameter tuning
    """
    logger.info("Starting hyperparameter tuning")
    
    # create parameter grid
    param_grid = {}
    for param_name, param_values in tuning_params.items():
        param_grid[param_name] = param_values
    
    # generate all combinations
    grid = list(ParameterGrid(param_grid))
    logger.info(f"Generated {len(grid)} parameter combinations")
    
    # train models with each parameter combination
    results = []
    for i, params in enumerate(grid):
        run_params = model_params.copy()
        
        # update parameters based on grid
        for param_name, param_value in params.items():
            if param_name in run_params:
                run_params[param_name] = param_value
            else:
                logger.warning(f"Parameter {param_name} not found in model params")
        
        logger.info(f"Training model {i+1}/{len(grid)} with params: {params}")
        result = train_single_model(run_params, X_train, y_train, X_val, y_val, seed, f"tuning_{i}")
        results.append(result)
    
    # find the best model
    best_result = max(results, key=lambda x: x['val_accuracy'])
    logger.info(f"Best model: {best_result['params']} with validation accuracy: {best_result['val_accuracy']:.4f}")
    
    # copy the best model to the output path
    shutil.copy(best_result['model_path'], "models/best_model.keras")
    
    # save the tuning results
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/training_metrics.json", 'w') as f:
        json.dump({
            'tuning_results': results,
            'best_model': best_result
        }, f, indent=2)
    
    return best_result

def main():
    parser = argparse.ArgumentParser(description="Train a CNN model for CIFAR-10 classification")
    parser.add_argument("--seed", type=int, default=26, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    params = load_params()
    model_params = params['model']
    
    # load data
    X_train, y_train, X_val, y_val, num_classes = load_data()
    
    # check if hyperparameter tuning is enabled
    if params['hyperparameter_tuning']['enabled']:
        tuning_params = params['hyperparameter_tuning']['parameters']
        best_model = hyperparameter_tuning(model_params, X_train, y_train, X_val, y_val, tuning_params, args.seed)
    else:
        # train a single model with the specified parameters
        result = train_single_model(model_params, X_train, y_train, X_val, y_val, args.seed, "baseline")
        
        # copy the model to the output path
        shutil.copy(result['model_path'], "models/best_model.keras")
        
        # save metrics
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/training_metrics.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    logger.info("Model training completed")

if __name__ == "__main__":
    import shutil
    main()