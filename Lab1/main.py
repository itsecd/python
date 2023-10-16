import os
import logging
import requests
import argparse
from bs4 import BeautifulSoup

BASE_URL = "https://www.bing.com/images/search"
HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
def create_directory(directory: str) -> None:
    """The function creates a folder if it does not exist"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as exc:
        logging.exception(f"Can't create folder: {exc.message}\n{exc.args}\n")


def download_img(search_query: str, num_images: int, base_url: str,
                     headers: str,) -> None:
    """This function uploads images to the selected directory"""
    create_directory("dataset")
    count = 0
    start = 1
    while count < num_images:
        base_url = f"{BASE_URL}?q={search_query}&form=HDRSC2&first={start}"
        response = requests.get(base_url, headers=headers)
        soup = BeautifulSoup(response.text, "lxml")
        img_tags = soup.find_all('img', {"src": True}, class_='mimg')
        for img in img_tags:
            img_url = img["src"]
            if img_url:
                try:
                    img_data = requests.get(img_url, headers=headers, timeout=10).content
                    class_folder = f"dataset/{search_query}"
                    create_directory(class_folder)
                    with open(f"{class_folder}/{count:04}.jpg", "wb") as img_file:
                        img_file.write(img_data)
                    count += 1
                    if count >= num_images:
                        break
                except Exception as e:
                    logging.exception(f"Error downloading image: {e.args}\n")
        start += 1
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download images in classes')
    parser.add_argument('--class1', type=str,default="polar_bear", help='name of first class')
    parser.add_argument('--class2', type=str,default="brown_bear", help='name of second class')
    parser.add_argument('--num_images', type=int,default=1000,help='num_images_per_class')
    args = parser.parse_args()
    download_img(args.class1, args.num_images, BASE_URL, HEADERS)
    download_img(args.class2, args.num_images, BASE_URL, HEADERS)