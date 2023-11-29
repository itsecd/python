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
        logging.error(f"Couldn't create folder: {ex.message}\n{ex.args}\n")


def make_list(url: str) -> list:
    '''принимает ссылку на запрос'''
    list_url = []
    try:
        for pages in range(main["pages"]):
            url_new = url[:-1]
            url_pages: str = f"{url_new}{pages}"
            html = requests.get(url_pages, main['headers'])
            soup = BeautifulSoup(html.text, "lxml")
            animals = soup.findAll("img")
            list_url += animals
        return list_url
    except Exception as ex:
        logging.error(f"List don't create: \n")


def download(
    max_files: int,
    classes: str,
    url: str,
    main_folder: str,
) -> str:
    ''' принимает имя папки, классы, URL и количество файлов'''
    count = 0
    except_count = 0
    for c in classes:
        url_list = make_list(url.replace("classes", c))
        for link in url_list:
            count_files = len(os.listdir(os.path.join(main_folder, c)))
            if count_files > max_files:
                count = 0
                continue
            try:
                src = link["src"]
                print(src)
                response = requests.get(src)
                create_directory(os.path.join(
                    main_folder, c).replace("\\", "/"))
                try:
                    # 
                    with open(os.path.join(main_folder, c, f"{count:04}.jpg").replace("\\", "/"), "wb") as file:
                        print(os.path.splitext(os.path.abspath("dataset/cat"))[1])
                        file.write(response.content)
                        count += 1
                except Exception as ex:
                    logging.error(f"Uncorrect path: {ex.message}\n{ex.args}\n")
            except Exception as ex:
                except_count += 1
                logging.error(
                    f"Quantity uncorrect URL={except_count}:{src}\n")
        logging.info(
            f"Quantity downloaded files in {c} class is {count_files}")


if __name__ == "__main__":
    with open(os.path.join("Lab1", "main.json"), "r") as file:
        main = json.load(file)

    download(main["max_files"], main["classes"], main["search_url"], main["main_folder"])