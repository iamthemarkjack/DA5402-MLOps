import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyspark import SparkContext, SparkConf
import findspark

# Set up Logging
os.makedirs('./logs', exist_ok=True) # Initialize Logs directory
logging.basicConfig(
    filename='./logs/task2.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

logger.info("Task 2 Started.")

# Stop any existing Spark Context
try:
    sc = SparkContext.getOrCreate()
    if sc._jsc is not None:
        sc.stop()
except Exception as e:
    logging.warning(f"Error stopping existing SparkContext: {e}")

# Start findspark and create a new context
findspark.init()

# Start new Spark Context
conf = SparkConf().setAppName("SentimentMetricsSpark").setMaster("local[4]")
sc = SparkContext(conf=conf)

def load_predictions(prediction_path):
    """
    Load predicted sentiment results from file.
    """
    try:
        with open(prediction_path, "r") as f:
            raw_data = f.read().strip().split("\n")
        parsed_data = sc.parallelize(raw_data, numSlices=10).map(lambda x: x.split("\t"))
        logging.info(f"Loaded {len(raw_data)} predictions from {prediction_path}")
        return parsed_data
    except Exception as e:
        logging.error(f"Failed to load predictions: {e}")
        return sc.parallelize([])
      
def compute_metrics(predictions_rdd):
    """
    Compute precision, recall and confusion matrix.
    """
    try:
        tp = predictions_rdd.filter(lambda x: float(x[1]) >= 3 and x[3] == 'positive').count()
        fp = predictions_rdd.filter(lambda x: float(x[1]) < 3 and x[3] == 'positive').count()
        tn = predictions_rdd.filter(lambda x: float(x[1]) < 3 and x[3] == 'negative').count()
        fn = predictions_rdd.filter(lambda x: float(x[1]) >= 3 and x[3] == 'negative').count()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        total = tp + fp + tn + fn

        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")

        # Normalized confusion matrix
        matrix = np.array([[tp / total, fn / total],
                           [fp / total, tn / total]])

        return matrix
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return np.zeros((2, 2))
    
def plot_confusion_matrix(matrix, save_path):
    """
    Plot and save the confusion matrix heatmap.
    """
    try:
        labels = ['Positive', 'Negative']
        percent_labels = np.vectorize(lambda x: f"{x * 100:.2f}%")(matrix)

        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=percent_labels, fmt='', cmap='Greens', linewidths=0.3,
                         linecolor='gray', cbar=False, square=True,
                         xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
        
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        logging.info(f"Confusion matrix saved at: {save_path}")
    except Exception as e:
        logging.error(f"Failed to generate confusion matrix: {e}")

if __name__ == "__main__":
    prediction_file_path = os.path.join(os.getcwd(), "outputs/task1_predictions.txt")
    confusion_matrix_path = os.path.join(os.getcwd(), "outputs/task2_confusion_matrix.png")

    predictions_rdd = load_predictions(prediction_file_path)
    matrix = compute_metrics(predictions_rdd)
    plot_confusion_matrix(matrix, confusion_matrix_path)