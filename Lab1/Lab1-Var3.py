import os
import requests
import logging
from bs4 import BeautifulSoup
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory(directory: str) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as exc:
        logger.exception(f"Can't create folder: {exc}") 


def download_images(query: str, 
                    num_images: int) -> None:
    create_directory("dataset")
    count = 0
    page = 1
    while count < num_images:
        search_url = f"https://www.bing.com/images/search?q={query}&form=HDRSC2&first={page}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "lxml")
        img_tags = soup.find_all('img', {"src": True}, class_='mimg')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for downloading images.")
    parser.add_argument("classes", nargs='+', help="List of classes to download images for.")
    parser.add_argument("--num_images_per_class", type=int, default=1000, help="Number of images to download per class.")
    args = parser.parse_args()

    for class_name in args.classes:
        download_images(class_name, args.num_images_per_class)
