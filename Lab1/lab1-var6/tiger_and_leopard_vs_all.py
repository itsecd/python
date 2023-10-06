"""Module providing a function printing python version 3.11.5."""
import os
from time import sleep
import logging
import requests
from bs4 import BeautifulSoup
from fake_headers import Headers


logging.basicConfig(level=logging.DEBUG)


def create_folders(name:str, base_folder: str = "dataset") -> None:
    """Form a folder"""
    try:
        if not os.path.exists(base_folder):
            os.makedirs(os.path.join(base_folder, name))
        elif  not os.path.exists(os.path.join(base_folder, name)):
            os.mkdir(os.path.join(base_folder, name))
    except OSError as e:
        logging.exception(f"OS error: {e}")


def make_path_and_filename(index: int, path_to_file: str) -> str:
    """This func creates the path to the future file and it's name"""
    filename = f"{index:04d}.jpg"
    return os.path.join(path_to_file, filename)


def save_image(url: str, filename: str) -> None:
    """This func downloads the image from the link"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.ok:
            with open(filename, 'wb') as file:
                file.write(response.content)
    except requests.exceptions.RequestException as e:
        logging.exception(f"Unable to download image: {url}:{e}")
    except Exception as e:
        logging.exception(f"Unable to download image: {url}:{e}")


def yandex_images_parser(text: str, base_folder: str = "dataset", url: str = "https://www.yandex.ru/images/search?text=") -> []:
    """parser 'Yandex.Images'"""
    create_folders(text)
    iterator = len(os.listdir(os.path.join(base_folder, text)))
    for page in range(iterator // 30, 40):
        headers = Headers(
            browser = "chrome",
            os = "win",
            headers = True
            ).generate()
        result = requests.get(f"{url}{text}&p={page}", headers, timeout= 10)
        logging.info(f"Page code received: {result.ok}")
        soup = BeautifulSoup(result.content, features = "lxml")
        links = soup.findAll("img",
                            class_ = "serp-item__thumb justifier__thumb"
                            )
        logging.info(f"links: {len(links)}")
        if len(links) == 0:
            logging.debug(soup.text)
            break
        for second_iterator in range(iterator % 30, len(links)):
            try:
                link = links[second_iterator].get("src")
                path_to_file = make_path_and_filename(iterator, os.path.join(base_folder,text))
                save_image(f"http:{link}",
                        path_to_file
                        )
                logging.info(f"Number of downloaded images: {iterator}")
                sleep(30)
                iterator += 1
            except Exception as e:
                logging.exception(f"Error with {iterator} image: {e}")
                continue


if __name__ == "__main__":
    yandex_images_parser("tiger")
    yandex_images_parser("leopard")
