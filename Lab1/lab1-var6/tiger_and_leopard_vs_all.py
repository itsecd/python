"""Module providing a function printing python version 3.11.5."""
import os
from time import sleep
import logging
import requests
from bs4 import BeautifulSoup
from fake_headers import Headers


logging.basicConfig(level=logging.DEBUG)

def create_folders(name:str) -> None:
    """This function create folders"""
    try:
        if not os.path.exists("dataset"):
            os.makedirs(os.path.join("dataset", name))
        elif  not os.path.exists(os.path.join("dataset", name)):
            os.mkdir(os.path.join("dataset", name))
    except OSError as err:
        logging.exception("OS error: %s",err)

def make_path_and_filename(index: int, path_to_file: str) -> str:
    """This func creates the path to the future file and it's name"""
    filename = f'{index:04d}' + ".jpg"
    return os.path.join("dataset", path_to_file, filename)

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

def yandex_images_parser(text : str, url: str) -> []:
    """parser 'Yandex.Images'"""
    create_folders(text)
    iterator = len(os.listdir(os.path.join("dataset", text)))
    for page in range(iterator//30, 40):
        headers = Headers(
            browser="chrome",
            os="win",
            headers=True
            ).generate()
        result = requests.get(url + f"&p={page}", headers, timeout= 10)
        logging.info("Page code received: %s", result.ok)
        soup = BeautifulSoup(result.content, features = "lxml")
        links = soup.findAll("img",
                            class_ = "serp-item__thumb justifier__thumb"
                            )
        logging.info("links: %d", len(links))
        if len(links) == 0:
            logging.debug(soup.text)
            break
        for second_iterator in range(iterator % 30, len(links)):
            try:
                link = links[second_iterator].get("src")
                path_to_file = make_path_and_filename(iterator, text)
                save_image("http:" + link,
                        path_to_file
                        )
                logging.info('Number of downloaded images: %d', iterator)
                sleep(30)
                iterator += 1
            except Exception:
                logging.exception('Error with %d image', iterator)
                continue

if __name__ == "__main__":
    yandex_images_parser("tiger", "https://www.yandex.ru/images/search?text=tiger")
    yandex_images_parser("leopard", "https://yandex.ru/images/search?from=tabbar&text=leopard")
