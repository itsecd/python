"""Module providing a function printing python version 3.11.5."""
import os
from time import sleep
import logging
import requests
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO)

def create_folders(name:str) -> None:
    """This function create a folder"""
    try:
        if not os.path.exists("dataset"):
            os.makedirs(os.path.join("dataset", name))
        elif  not os.path.exists(os.path.join("dataset", name)):
            os.mkdir(os.path.join("dataset", name))
    except OSError as err:
        logging.exception("OS error: %s",err)

def make_path_and_filename(index: int, path: str) -> str:
    """This func creates the path to the future file and it's name"""
    filename = f'{index:04d}' + ".jpg"
    return os.path.join("dataset", path, filename)

def save_image(url: str, filename: str) -> None:
    """This func downloads the image from the link"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.ok:
            with open(filename, 'wb') as file:
                file.write(response.content)
    except requests.exceptions.RequestException:
        logging.exception('Unable to download image: %s %s %s',
              url,
              ':',
              str(requests.exceptions.RequestException)
              )
    except Exception:
        logging.exception('Unable to download image: %s %s %s',
              url,
              ':',
              str(Exception)
              )

def yandex_images_iarser(text : str, url: str) -> []:
    """parser 'Yandex.Images'"""
    create_folders(text)
    i = 0
    for page in range(20):
        headers = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/118.0"
        result = requests.get(url + f"&p={page}", headers, timeout= 10)
        logging.info("Page code received: %s", result.ok)
        soup = BeautifulSoup(result.content, features = "lxml")
        links = soup.findAll("img",
                            class_ = "serp-item__thumb justifier__thumb"
                            )
        logging.info("links: %d", len(links))
        for link in links:
            try:
                link = link.get("src")
                path_to_file = make_path_and_filename(i, text)
                if os.path.exists(path_to_file):
                    continue
                save_image("http:" + link,
                        path_to_file
                        )
                i += 1
                logging.info('Image download: %d', i)
                sleep(3)
            except Exception:
                logging.exception('Error with %d image', i)
                continue

if __name__ == "__main__":
    yandex_images_iarser("tiger", "https://yandex.ru/images/search?text=tiger")
    yandex_images_iarser("leopard", "https://yandex.ru/images/search?from=tabbar&text=leopard")
