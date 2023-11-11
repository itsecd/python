import os
import json
import logging
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

def create_directory(folder: str) -> str:
    '''принимает путь к папке и ее имя'''
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except Exception as ex:
        logging.error(f"Error in create_directory")


def make_list(url: str) -> list:
    '''принимает ссылку на запрос'''
    list_url = []
    try:
        for pages in range(main["pages"]):
            url_new = url[:-1]
            url_pages: str = f"{url_new}{pages}"
            # делаем запрос и получаем html
            responce = requests.get(url_pages, main['headers']).text
            soup = BeautifulSoup(responce, "lxml")  # используем парсер lxml
            animals = soup.findAll("img")
            list_url += animals
        return list_url
    except Exception as ex:
        logging.error(f"Error in make_list")


def download(
    folder: str,
    search: str,
    url: str,
    max_files: int,
) -> str:
    ''' принимает имя папки, классы, URL и количество файлов'''
    count = 0
    incorrect_url = 0
    for cd in search:
        url_list = make_list(url.replace("search", cd))
        for exile in url_list:
            total_files = len(os.listdir(os.path.join(folder, cd)))
            if total_files > max_files:
                continue
            try:
                src = exile["src"]
                response = requests.get(src)
                create_directory(os.path.join(folder, cd))
                try:
                    with open(os.path.join(folder, cd, f"{count:04}.jpg"), "wb") as file:
                        file.write(response.content)
                        count += 1
                except Exception as ex:
                    logging.error(f"Incorrect path: {ex}")
            except Exception as ex:
                incorrect_url += 1
                logging.error(f"Total incorrect URL: {incorrect_url}")
    logging.info(f"Completoin of data loading")


if __name__ == "__main__":
    with open(os.path.join("Lab1", "main.json"), "r") as main_file:
        main = json.load(main_file)

    download(main["folder"], main["search"], main["url"], main["max_file"])
