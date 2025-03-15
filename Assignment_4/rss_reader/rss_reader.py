import os
import sys
import time
import requests
import logging
import datetime
import hashlib
import feedparser
import pymongo
from pymongo.errors import DuplicateKeyError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('rss_reader')

# Environment variables with defaults
MONGO_HOST = os.environ.get('MONGO_HOST', 'mongodb')
MONGO_PORT = int(os.environ.get('MONGO_PORT', '27017'))
MONGO_DB = os.environ.get('MONGO_DB', 'newsdb')
MONGO_USER = os.environ.get('MONGO_APP_USERNAME', 'newsapp')
MONGO_PASSWORD = os.environ.get('MONGO_APP_PASSWORD', 'newsapp_password')
RSS_FEED_URL = os.environ.get('RSS_FEED_URL', 'https://www.thehindu.com/news/national/feeder/default.rss')
RSS_SOURCE = os.environ.get('RSS_SOURCE', 'The Hindu')
TITLE_PATH = os.environ.get('TITLE_PATH', 'title')
LINK_PATH = os.environ.get('LINK_PATH', 'link')
SUMMARY_PATH = os.environ.get('SUMMARY_PATH', 'summary')
DATE_PATH = os.environ.get('DATE_PATH', 'published_parsed')
IMAGE_URL_PATH = os.environ.get('IMAGE_URL_PATH', 'media_content/0/url')
POLL_INTERVAL = int(os.environ.get('POLL_INTERVAL', '600'))  # 10 minutes in seconds
STARTUP_DELAY = int(os.environ.get('MONGODB_STARTUP_DELAY', '7'))  # Seconds to wait for MongoDB to initialize
MAX_RETRIES = 30
RETRY_INTERVAL = 10

def wait_for_mongodb():
    """Wait for MongoDB to be available with retries"""
    # Initial delay to let MongoDB initialize fully
    logger.info(f"Waiting {STARTUP_DELAY} seconds for MongoDB to initialize...")
    time.sleep(STARTUP_DELAY)
    
    logger.info(f"Attempting to connect to MongoDB at {MONGO_HOST}:{MONGO_PORT}")
    for attempt in range(MAX_RETRIES):
        try:
            # Connection string with authentication
            connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}?authSource={MONGO_DB}"
            client = pymongo.MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB on attempt {attempt+1}")
            return client
        except Exception as e:
            logger.warning(f"Connection attempt {attempt+1}/{MAX_RETRIES} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_INTERVAL} seconds...")
                time.sleep(RETRY_INTERVAL)
            else:
                logger.error("Max retries reached. Cannot connect to MongoDB.")
                raise

def get_mongodb_client():
    """Establish connection to MongoDB with retries"""
    try:
        client = wait_for_mongodb()
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB after multiple attempts: {str(e)}")
        sys.exit(1)

def get_feed_etag(feed_url):
    """Get current ETag or Last-Modified from feed URL"""
    try:
        response = requests.get(feed_url)
        etag = response.headers.get('ETag')
        last_modified = response.headers.get('Last-Modified')
        return etag or last_modified
    except Exception as e:
        logger.warning(f"Failed to get feed headers: {str(e)}")
        return None

def parse_feed(feed_url):
    """Parse the RSS feed and return entries"""
    logger.info(f"Fetching feed from {feed_url}")
    feed = feedparser.parse(feed_url)
    
    if hasattr(feed, 'status') and feed.status != 200:
        logger.warning(f"Failed to fetch feed: Status {feed.status}")
        return []
    
    logger.info(f"Successfully fetched feed with {len(feed.entries)} entries")
    return feed.entries

def get_value(entry, path):
    """Get the value in the given path of a single feed entry"""
    path_components = [int(a) if a.isnumeric() else a for a in path.split('/')]
    try:
        value = entry
        for component in path_components:
            if isinstance(value, list) and isinstance(component, int) and component < len(value):
                value = value[component]
            elif isinstance(value, dict) and component in value:
                value = value[component]
            else:
                return None
        return value
    except Exception as e:
        logger.error(f"Error in accessing the path {path}: {e}")
        return None

def process_entry(entry):
    """Process a single feed entry"""
    source = RSS_SOURCE
    title = get_value(entry, TITLE_PATH)
    link = get_value(entry, LINK_PATH)
    summary = get_value(entry, SUMMARY_PATH)
    pub_date = get_value(entry, DATE_PATH)
    image_url = get_value(entry, IMAGE_URL_PATH)

    # Handle date conversion
    if pub_date:
        if isinstance(pub_date, time.struct_time):
            pub_date = datetime.datetime(*pub_date[:6])
        else:
            try:
                pub_date = datetime.datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z')
            except Exception as e:
                logger.warning(f"Could not parse date: {e}")
                pub_date = datetime.datetime.now()
    else:
        pub_date = datetime.datetime.now()

    # Download the image content if URL exists
    image = None
    if image_url:
        try:
            response = requests.get(image_url)
            if response.status_code == 200 and response.content:
                image = response.content
            else:
                logger.warning(f"Failed to retrieve image for {title}: Status {response.status_code}")
        except Exception as e:
            logger.warning(f"Error downloading image: {e}")
    
    article = {
        'title': title,
        'link': link,
        'image': image,
        'image_url': image_url,
        'pubDate': pub_date,
        'source': source,
        'summary': summary
    }

    return article

def save_to_mongodb(articles, client):
    """Save articles to MongoDB"""
    db = client[MONGO_DB]
    collection = db.articles
    
    inserted_count = 0
    for article in articles:
        try:
            # Using upsert to prevent duplicates based on the unique index
            result = collection.update_one(
                {'link': article['link']},
                {'$set': article},
                upsert=True
            )
            if result.upserted_id:
                inserted_count += 1
                logger.debug(f"Inserted new article: {article['title']}")
        except DuplicateKeyError:
            logger.debug(f"Duplicate article detected: {article['title']}")
        except Exception as e:
            logger.error(f"Error saving article {article['title']}: {str(e)}")
    
    return inserted_count

def main():
    """Main function to poll RSS feeds and save to MongoDB"""
    logger.info("Starting RSS Feed Reader Application")
    logger.info(f"Configuration: Feed URL={RSS_FEED_URL}, Source={RSS_SOURCE}, Poll Interval={POLL_INTERVAL}s")
    
    # Connect to MongoDB
    client = get_mongodb_client()
    
    # Initialize last_etag
    last_etag = None
    
    while True:
        try:
            # Check if feed has changed
            current_etag = get_feed_etag(RSS_FEED_URL)
            if current_etag != last_etag:
                logger.info("Feed has changed, processing new content")
                
                # Parse feed
                entries = parse_feed(RSS_FEED_URL)
                # Process entries
                articles = [process_entry(entry) for entry in entries]
                # Save to MongoDB
                inserted_count = save_to_mongodb(articles, client)
                logger.info(f"Processed {len(articles)} articles, inserted {inserted_count} new articles")
                # Update etag
                last_etag = current_etag
            else:
                logger.info("No new updates in feed")

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        
        logger.info(f"Sleeping for {POLL_INTERVAL} seconds")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()