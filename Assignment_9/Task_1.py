# Script for Task 1
import os
import re
import logging
from pyspark import SparkContext, SparkConf
from transformers import pipeline
import findspark

# Set up Logging
os.makedirs('./logs', exist_ok=True) # Initialize Logs directory
logging.basicConfig(
    filename='./logs/task1.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

logger.info("Task 1 Started.")

# Stop any existing Spark Context
try:
    sc = SparkContext.getOrCreate()
    if sc._jsc is not None:
        sc.stop()
except Exception as e:
    logging.warning(f"Error stopping existing SparkContext: {e}")

# Start findspark and create a new context
findspark.init()

def load_reviews(file_path):
    """
    Reads the review file and splits it into individual review blocks.
    """
    try:
        with open(file_path, 'r') as file:
            reviews = file.read().split("\n\n")
        logging.info(f"Loaded {len(reviews)} reviews from {file_path}")
        return reviews
    except Exception as e:
        logging.error(f"Failed to read file {file_path}: {e}")
        return []
    
def parse_review(raw_text):
    """
    Extract product ID, rating, and review text from a single raw review.
    """
    product_id = re.search(r'product/productId: (.+)', raw_text)
    rating = re.search(r'review/score: (.+)', raw_text)
    text = re.search(r'review/text: (.+)', raw_text)

    print(product_id, rating, text)

    return (
        product_id.group(1) if product_id else "",
        float(rating.group(1)) if rating else 0.0,
        text.group(1) if text else ""
    )

def sentiment_mapper(records):
    """
    Sentiment analysis function to be used with Spark's mapPartitions.
    Loads the transformer pipeline inside each partition for efficiency.
    """
    sentiment_pipeline = pipeline("sentiment-analysis", model="sentence-transformers/all-MiniLM-L6-v2")

    for record in records:
        product_id, rating, review = record
        try:
            prediction = sentiment_pipeline(review[:512])[0]
            label = prediction['label'].lower()
            if label == 'label_0':
                label = 'negative'
            elif label == 'label_1':
                label = 'positive'
        except Exception as e:
            logging.error(f"Sentiment analysis failed for review: {review[:30]}... Error: {e}")
            label = "nil"

        yield (product_id, rating, review, label)
    
if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data/Gourmet_Foods.txt")
    raw_reviews = [load_reviews(data_path)[0]]
    
    # Create Spark Context
    conf = SparkConf().setAppName("SparkSentimentAnalysis").setMaster("local[4]")
    sc = SparkContext(conf=conf) 

    # Parallelize raw review data
    rdd_reviews = sc.parallelize(raw_reviews)

    # Clean and filter raw data
    parsed_reviews = rdd_reviews.map(parse_review).filter(
                        lambda x: x[0] != "" and x[1] >= 0.0 and x[2] != ""
                    )

    results_rdd = parsed_reviews.mapPartitions(sentiment_mapper)
    results = results_rdd.collect()
    
    # Write output to file
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "task1_predictions.txt")

    try:
        with open(output_path, "w") as f:
            for record in results:
                f.write("\t".join(map(str, record)) + "\n")
        logging.info(f"Results written successfully to {output_path}")
    except Exception as e:
        logging.error(f"Failed to write results: {e}")