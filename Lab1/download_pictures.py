import os
import requests
import logging
import json
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO)


def create_folder(folder_name: str) -> None:
    """The function takes folder name and create folder,  if it does not exist"""
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except Exception as ex:
        logging.error(f"Failed to create folder:{ex.message}\n{ex.args}\n")


def get_images_urls(queries: str,
                    count: int,
                    url: str
                    ) -> str:
    """The function takes three parameters: queries (str): the search query to be used in the URL, 
    count (int): the number of image URLs to retrieve, url (str): the base URL to be used for retrieving 
    the images.The function searches query and creates str: a list of image urls.
    """

    urls = []
    page = 1
    while len(urls) < count:
        url_new = url.replace("{query}", queries)
        url1 = url_new.replace("{page}", str(page))
        response = requests.get(url1)
        soup = BeautifulSoup(response.text, "html.parser")
        images = soup.find_all("img")
        for image in images:
            image_url = image.get("src")
            if image_url and image_url.startswith("https://"):
                urls.append(image_url)
        page += 1
    return urls[:count]


def dowload_images(queries: str,
                   count: int,
                   url: str,
                   main_folder: str
                   ) -> None:
    """The function takes parameters: queries (str), count (int), url (str), main_folder (str): 
    The main folder where the images will be saved. Downloads images from specified urls and saves 
    them to specified folders.
    """

    except_count = 0
    queries = options["queries"]
    for query in queries:
        image_urls = get_images_urls(query, count, url)
        create_folder(os.path.join(main_folder, query).replace("\\", "/"))
        try:
            for i, url1 in enumerate(image_urls):
                response = requests.get(url1)
                if response.status_code == 200:
                    try:
                        with open(os.path.join(main_folder, query, f"{i:04d}.jpg").replace("\\", "/"), "wb") as file:
                            file.write(response.content)
                    except Exception as ex:
                        logging.error(
                            f"Uncorrect path:{ex.message}\n{ex.args}\n")
        except Exception as ex:
            except_count += 1
            logging.error(f"Quantity uncorrect URl={except_count}:{url1}\n")


if __name__ == "__main__":
    with open(os.path.join("options.json"), "r") as options_file:
        options = json.load(options_file)

    dowload_images(options["queries"], options["count"],
                   options["url"], options["main_folder"])
