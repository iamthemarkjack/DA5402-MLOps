import logging
import json
import requests
from bs4 import BeautifulSoup
from selenium.webdriver import Chrome

def download_data(url, home_url):
    """
    Downloads the image and meta-data from the Top stories page
    """
    data = []

    # initiating the chrome driver and pull the html text from the lazy loading page
    driver = Chrome()
    print("Opening the Top Stories page in Google Chrome....")
    driver.get(url)

    soup = BeautifulSoup(driver.page_source, 'lxml')
    print(f"Successfully fetched the html text of {url}")
    logging.info(f"Successfully fetched the html text of {url}")

    articles_with_figure = [article for article in soup.find_all("article") if article.find("figure")]

    for article in articles_with_figure:
        caption = article.find_all('a')[-1].text
        datetime = article.find('time')['datetime']
        if caption:
            rel_img_src = article.find('img')['src']
            # modifying the img src url to get better resolution image originally it will be 280x168
            rel_img_src = rel_img_src.replace("w280-h168","w1920-h1080")
            img_src = home_url + rel_img_src[1:]
            response = requests.get(img_src)
            if response.status_code == 200 and response.content:
                data.append([caption,img_src,datetime,response.content])
                logging.info(f"Image downloaded successfully for headline: {caption}")
            else:
                logging.error(f"Failed to retrieve the image for headline: {caption}")

    return data