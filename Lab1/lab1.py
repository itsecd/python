import os
import requests
import logging
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO)


def create_folder(folder_name: str) -> None:
    """The function takes folder name and create folder, if it is't being """
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except Exception as ex:
        logging.error(f"Failed to create folder:{ex.message}\n{ex.args}\n")


def get_images_urls(query: str, 
                    count: int
                    ) -> str:
    """The function searches query, creates a list of query's urls"""
    urls = []
    page = 1
    while len(urls) < count:
        url = f"https://www.bing.com/images/search?q={query}&form=hdrsc2&first={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        images = soup.find_all("img")
        for image in images:
            image_url = image.get("src")
            if image_url and image_url.startswith("https://"):
                urls.append(image_url)
        page += 1
    return urls[:count]

def dowload_image(image_urls: str,
                  folder_name: str
                  ) -> None:
    """The functions takes image's URLS, dowloads pictures and adds them to folder"""
    except_count = 0
    try:
        for i, url in enumerate(image_urls):
            response = requests.get(url)
            if response.status_code == 200:
                file_name = f"dataset/{folder_name}/{i:04d}.jpg"
            try:
                with open(file_name, "wb") as file:
                    file.write(response.content)
            except Exception as ex:
                logging.error(f"Uncorrect path:{ex.message}\n{ex.args}\n")
    except Exception as ex:
        except_count +=1
        logging.error(f"Quantity uncorrect URl={except_count}:{url}\n")


queries = ["tiger", "leopard"]
create_folder("dataset")
for query in queries:
    folder_name = f"dataset/{query}"
    create_folder(folder_name)
tiger_urls = get_images_urls("tiger", 50)
dowload_image(tiger_urls, "tiger")
leopard_urls = get_images_urls("leopard", 50)
dowload_image(leopard_urls, "leopard")
