import os
from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
import datetime
from bson.binary import Binary
import base64

app = Flask(__name__)

# Environment variables
MONGO_HOST = os.environ.get('MONGO_HOST', 'mongodb')
MONGO_PORT = int(os.environ.get('MONGO_PORT', '27017'))
MONGO_DB = os.environ.get('MONGO_DB', 'newsdb')
MONGO_USER = os.environ.get('MONGO_APP_USERNAME', 'newsapp')
MONGO_PASSWORD = os.environ.get('MONGO_APP_PASSWORD', 'newsapp_password')

def get_mongodb_client():
    """Establish connection to MongoDB"""
    try:
        connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}?authSource={MONGO_DB}"
        client = MongoClient(connection_string)
        # Test connection
        client.admin.command('ping')
        app.logger.info(f"Successfully connected to MongoDB")
        return client
    except Exception as e:
        app.logger.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

@app.route('/')
def index():    
    # Connect to MongoDB
    client = get_mongodb_client()
    
    # Query articles based on date filter
    db = client[MONGO_DB]
    articles_collection = db.articles

    # Sort by publication date (newest first), limit to the latest 100
    articles = articles_collection.find().sort('pubDate', -1).limit(100)
    
    # Process articles
    processed_articles = []
    for article in articles:
        processed_articles.append({
            'title': article.get('title', 'Untitled'),
            'link': article.get('link', '#'),
            'summary': article.get('summary', 'No summary available'),
            'image_url': article.get('image_url', '#'),
            'source': article.get('source', '#')
        })
    
    return render_template('index.html', articles=processed_articles)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)