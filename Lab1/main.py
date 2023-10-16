import os
import logging
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.bing.com/images/search"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
NUM_IMAGES_PER_CLASS = 5
def create_directory(directory: str) -> None:
    """The function creates a folder if it does not exist"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as exc:
        logging.exception(f"Can't create folder: {exc.message}\n{exc.args}\n")


def download_img(search_query: str, num_images: int) -> None:
    """This function uploads images to the selected directory"""
    create_directory("dataset")
    count = 0
    start = 0
    while count < num_images:
        base_url = f"{BASE_URL}?q={search_query}&form=HDRSC2&first={start}"
        headers = {
            "User-Agent": USER_AGENT,
        }
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
                    logging.exception(f"Error downloading image: {e.message}\n{e.args}\n")
        start += 1
    

if __name__ == "__main__":
    classes = ["polar bear", "brown bear"]
    for class_name in classes:
        download_img(class_name, NUM_IMAGES_PER_CLASS)