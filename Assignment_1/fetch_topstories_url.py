import logging
from bs4 import BeautifulSoup

def fetch_url(html_text, kws, home_url):
    """
    Finds Top stories url from Google News Home page html text
    """
    soup = BeautifulSoup(html_text,'lxml')
    url = None
    possible_sections = soup.find_all("a") 
    for possible_section in possible_sections:
        if possible_section.text.lower() in kws: # checking whether the text contains keywords
            url = home_url + possible_section['href'][2:] # appending the relative url to the home url
            break
    if url:
        print(f"Successfully retrieved the Top stories URL: {url}")
        logging.info(f"Successfully retrieved the Top stories URL: {url}")
        return url
    else:
        print("Failed to retrieve the Top stories URL")
        logging.error("Failed to retrieve the Top stories URL")