import requests
import logging

def fetch_homepage(home_url):
    """
    Returns the Google News home-page html when called
    """
    print(f"Trying to fetch the html text of {home_url}....")
    try:
        response = requests.get(home_url)
        html_text = response.text
        print("Successfully fetched Google News homepage")
        logging.info("Successfully fetched Google News homepage")
        return html_text
    except requests.RequestException as e:
        print(f"Error fetching Google News homepage: {e}")
        logging.error(f"Error fetching Google News homepage: {e}")
        return None
