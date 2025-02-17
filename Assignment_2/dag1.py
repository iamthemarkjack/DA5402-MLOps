import os
from datetime import datetime
import requests
import base64
from bs4 import BeautifulSoup
from selenium import webdriver
import json

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mongo.hooks.mongo import MongoHook

default_args = {
    "owner" : "Rohith Ramanan"
}

def fetch_homepage(home_url):
    """
    Returns the Google News home-page html when called
    """
    print(f"Trying to fetch the html text of {home_url}....")
    try:
        response = requests.get(home_url)
        html_text = response.text
        print("Successfully fetched Google News homepage")
        return html_text
    except requests.RequestException as e:
        print(f"Error fetching Google News homepage: {e}")
        return None

def fetch_url(ti, kws, home_url):
    """
    Finds Top stories url from Google News Home page html text
    """
    html_text = ti.xcom_pull(task_ids = "fetch_homepage")
    soup = BeautifulSoup(html_text,"lxml")
    url = None
    possible_sections = soup.find_all("a") 
    for possible_section in possible_sections:
        if possible_section.text.lower() in kws: # checking whether the text contains keywords
            url = home_url + possible_section['href'][2:] # appending the relative url to the home url
            break
    if url:
        print(f"Successfully retrieved the Top stories URL: {url}")
        return url
    else:
        print("Failed to retrieve the Top stories URL")

def download_data(ti, home_url, remote_webdriver='remote_chromedriver'):
    """
    Downloads the image and meta-data from the Top Stories page using Remote WebDriver.
    """
    url = ti.xcom_pull(task_ids="fetch_url")
    data = []

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

    # connect to remote WebDriver
    with webdriver.Remote(f'http://{remote_webdriver}:4444/wd/hub', options=options) as driver:
        print("Opening the Top Stories page in Google Chrome (Remote)...")
        driver.get(url)

        soup = BeautifulSoup(driver.page_source, 'lxml')
        print(f"Successfully fetched the html text of {url}")

        articles_with_figure = [article for article in soup.find_all("article") if article.find("figure")]

        for article in articles_with_figure:
            caption = article.find_all('a')[-1].text
            datetime = article.find('time')['datetime']
            if caption:
                rel_img_src = article.find('img')['src']
                # modify the image URL for better resolution
                rel_img_src = rel_img_src.replace("w280-h168", "w1920-h1080")
                img_src = home_url + rel_img_src[1:]

                response = requests.get(img_src)
                if response.status_code == 200 and response.content:
                    img_base64 = base64.b64encode(response.content).decode('utf-8')
                    data.append([caption, img_src, datetime, img_base64])

    return json.dumps(data)

def save_data(ti):
    """
    Saves the image data and the metadata in MongoDB Database after checking for duplicates
    """

    data_json = ti.xcom_pull(task_ids="download_data")
    data = json.loads(data_json)

    count = 0 # number of new inserts

    # setting up the MongoDB Client
    hook = MongoHook(mongo_conn_id='mongo_default')
    client = hook.get_conn()
    print(f"Connected to MongoDB - {client.server_info()}")

    db = client.image_captioning_database

    captions_collection = db.captions
    images_collection = db.images

    def is_duplicate(caption, datetime):
        """
        Check if a headline with the same caption and datetime already exists
        """
        return captions_collection.find_one({"caption": caption, "datetime": datetime}) is not None

    print("Trying to save the downloaded data in MongoDB Database")
    for caption, img_url, datetime, img_data in data:
        if is_duplicate(caption, datetime):
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
                "image_data" : img_data
            }

            images_collection.insert_one(image_doc)

            count += 1

            print(f"Successfully saved image for: {caption} in the database")
        
        except Exception as e:
            print(f"Failed to save image for: {caption}")

    status_path = "/opt/airflow/dags/run/status"
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path,"w") as f:
        f.write(str(count))

with DAG(
    default_args = default_args,
    dag_id = "DAG_1",
    description = "Downloads the data, stores them in MongoDB Database, and writes a status file",
    start_date = datetime(2025, 2 ,10),
    catchup = False,
    schedule_interval = "@hourly"
 ) as dag:
    task1 = PythonOperator(
        task_id = "fetch_homepage",
        python_callable = fetch_homepage,
        op_kwargs = {"home_url" : "https://news.google.com/"}
    )

    task2 = PythonOperator(
        task_id = "fetch_url",
        python_callable = fetch_url,
        op_kwargs = {"kws" : ["top stories","hot topics"],
                     "home_url" : "https://news.google.com/"}
    )

    task3 = PythonOperator(
        task_id = "download_data",
        python_callable = download_data,
        op_kwargs = {"home_url" : "https://news.google.com/"}
    )

    task4 = PythonOperator(
        task_id = "save_data",
        python_callable = save_data,
        op_kwargs = {"home_url" : "https://news.google.com/"}
    )

    task1 >> task2 >> task3 >> task4