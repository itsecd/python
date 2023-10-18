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
        logging.error(f"Error in create_directory") 

def make_list(url: str) -> list: # принимает ссылку на запрос
    list_url = []
    try:
        for pages in range(main["pages"]):
            url_new = url[:-1]
            url_pages: str = f"{url_new}{pages}"
            responce = requests.get(url_pages, main['headers']).text # делаем запрос и получаем html
            soup = BeautifulSoup(responce, "lxml") # используем парсер lxml
            animals= soup.findAll("img")
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
    count = 0
    incorrect_url = 0
    for cd in search:
        url_list = make_list(url.replace("search", cd)) # replace заменяет все вхождения подстроки "search" на cd
        for exile in url_list:
            total_files = len(os.listdir(os.path.join(folder, cd))) # len принимает объект в качестве аргумента и возвращает его длину; listdir возвращает список всех файлов в данном каталоге
            if total_files > max_files: continue
            try:
                src = exile["src"]
                response = requests.get(src)
                create_directory(os.path.join(folder, cd).replace("\\", "/"))
                try:
                    with open(os.path.join(folder, cd, f"{count:04}.jpg").replace("\\", "/"), "wb") as file:
                        file.write(response.content)
                        count += 1
                except Exception as ex:
                    logging.error(f"Incorrect path: {ex}")
            except Exception as ex:
                incorrect_url += 1
                logging.error(f"Total incorrect URL: {incorrect_url}")

if __name__ == "__main__":
    with open(os.path.join("Lab1", "main.json"), "r") as main_file: # объединяет компаненты пути
        main = json.load(main_file) # чтение json-данных из файла и преобразование их в словарь

    download(main["folder"], main["search"], main["url"], main["max_files"])
