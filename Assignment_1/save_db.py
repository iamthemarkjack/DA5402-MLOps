import os
import pymongo
import logging

# setting up the MongoDB Client
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["image_captioning_database"]

captions_collection = db["captions"]
images_collection = db["images"]

def is_duplicate(caption, datetime):
    """
    Check if a headline with the same caption and datetime already exists
    """
    return captions_collection.find_one({"caption": caption, "datetime": datetime}) is not None

def save_data(data, image_dir):
    """
    Saves the image data and the metadata in MongoDB Database after checking for duplicates
    """
    print("Trying to save the downloaded data in MongoDB Database")
    for caption, img_url, datetime, img_data in data:
        if is_duplicate(caption, datetime):
            logging.warning(f"Duplicate found, skipping: {caption}")
            continue

        try:
            metadata = {
                "caption": caption,
                "img_url": img_url,
                "datetime": datetime
            }
            
            metadata_id = captions_collection.insert_one(metadata).inserted_id

            image_doc = {
                "caption_id" : metadata_id,
                "image_data" : img_data,
                "format" : "jpeg"
            }

            images_collection.insert_one(image_doc)

            # to save the images as jpeg in the local with metadata_id as their name
            name = str(metadata_id) + ".jpg"
            with open(image_dir + name ,"wb") as file:
                file.write(img_data)

            logging.info(f"Successfully saved image for: {caption} in the database")
        
        except Exception as e:
            logging.error(f"Failed to save image for: {caption}")