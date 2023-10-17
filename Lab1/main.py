import os
import json
import logging
import os.path
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

def create_directory(folder: str) -> str: # принимает путь к папке и ее имя
    try:
        if not os.path.exists(folder):  # возвращает false, если путь не существует
            os.makedirs(folder) # создает промежуточные каталоги по пути folder, если они не существуют
    except Exception as ex:
        logging.error("Error in create_directory") 

def make_list(url: str) -> list: # принимает ссылку на запрос
    list_url = []
    try:
        for pages in range(main["pages"]):
            url_new = url[:-1]
            url_pages: str = f"{url_new}{pages}"
            responce = requests.get(url_pages, main['headers']).text
            soup = BeautifulSoup(responce, "lxml")
            animals= soup.findAll("img")
            list_url += animals
        return list_url
    except Exception as ex:
        logging.error("Error in make_list")

if __name__ == "__main__":
    with open(os.path.join("Lab1", "main.json"), "r") as main_file: # объединяет компаненты пути
        main = json.load(main_file) # чтение json-данных из файла и преобразование их в словарь

    download(main["max_files"], main["classes"], main["search_url"], main["main_folder"])
