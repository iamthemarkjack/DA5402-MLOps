import os
import argparse
import logging
import json
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/evaluate_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("evaluate_model")

def load_data():
    """
    Load test data
    """
    test_df = pd.read_pickle("data/test/test.pkl")
    
    X_test = np.stack(test_df['image_data'].values)
    y_test = test_df['class_idx'].values
    
    class_names = sorted(test_df['class_name'].unique())
    
    # normalize the data
    X_test = X_test.astype('float32') / 255.0
    
    # keep original labels for confusion matrix
    y_test_original = y_test.copy()
    
    # convert labels to one-hot encoding
    num_classes = len(np.unique(y_test))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    logger.info(f"Loaded test data: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")
    
    return X_test, y_test, y_test_original, class_names

def evaluate_model(model, X_test, y_test, y_test_original, class_names, seed):
    """
    Evaluate the model and generate performance metrics
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    logger.info(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
    
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # create confusion matrix
    cm = confusion_matrix(y_test_original, y_pred)
    
    # calculate per-class metrics
    class_report = classification_report(y_test_original, y_pred, target_names=class_names, output_dict=True)
    
    # calculate class-wise accuracy
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = (y_test_original == i)
        class_accuracy[class_name] = np.mean(y_pred[class_mask] == i)
    
    # create directories for metrics
    os.makedirs("metrics", exist_ok=True)
    
    # save metrics
    metrics = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'class_accuracy': class_accuracy,
        'classification_report': class_report
    }
    
    with open("metrics/evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # save class accuracies separately
    with open("metrics/class_accuracies.json", 'w') as f:
        json.dump(class_accuracy, f, indent=2)
    
    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("metrics/confusion_matrix.png")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN model for CIFAR-10 classification")
    parser.add_argument("--seed", type=int, default=26, help="Random seed")
    args = parser.parse_args()
    
    model_path = "models/best_model.keras"
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    X_test, y_test, y_test_original, class_names = load_data()
    
    metrics = evaluate_model(model, X_test, y_test, y_test_original, class_names, args.seed)
    
    logger.info("Model evaluation completed")

if __name__ == "__main__":
    main()