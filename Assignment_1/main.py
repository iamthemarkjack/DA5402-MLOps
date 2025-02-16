import os
import logging
from datetime import date
import json

from fetch_homepage import fetch_homepage
from fetch_topstories_url import fetch_url
from retrieve_data import download_data
from save_db import save_data

# loading the configuration file
with open("config.json","r") as f:
    config = json.load(f)

# getting the configuration variables
home_url = config["home_page_url"]
kws = config["keywords"]
image_dir = config["image_dir"]

# set up logging
os.makedirs("logs",exist_ok=True)
today = str(date.today())
logging.basicConfig(filename=f"logs/{today}.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# setting up the image directory for storing the images locally
os.makedirs(image_dir,exist_ok=True)

def main():
    # fetching the html text of the google news home page (MODULE-1)
    html_text = fetch_homepage(home_url)

    # retrieving the url for top stories page from the google news home page (MODULE-2)
    url = fetch_url(html_text, kws, home_url)

    # downloading the image data, caption and relevant meta-data from the top stories page (MODULE-3)
    data = download_data(url, home_url)

    # saving the data in a MongoDB Database and as well storing the images in local for reference (MODULE-4)
    # along with de-duplication constraint (MODULE-5)
    save_data(data, image_dir)


# orchestration (MODULE-6)
main()