import os
import time
import random
import logging
import requests
import json
from bs4 import BeautifulSoup

def create_folder(folder: str) -> None:
    """
    Creates a folder 

    Сreates a folder using the passed path
    Parameters
    ----------
    folder : str
        Путь для создания папки
    """
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except Exception as err: 
        logging.error(f"{err}", exc_info=True)


def download(list_images : list, folder : str, count_images : int) -> None:
    """
    Download images

    Download images from list
    Parameters
    ----------
    list_images : list
        Список тегов с картинками
    folder : str
        Путь для скачивания
    count : int
        Кол-во картинок для скачивания
    """
    count = 0
    exec_count = 0
    for flower_url in list_images:
        if count >= count_images : break
        try:
            src=flower_url['src']
            response = requests.get(src)
            create_folder(folder)
            with open(os.path.join(folder, f"{count:04}.jpg").replace("\\", "/"), "wb") as file:
                file.write(response.content)
                count+=1
        except Exception as err:
            exec_count+=1
            logging.warning(f"Error image_url {count+exec_count+1}. {err}")
    logging.info(f"Successful download - {count}")
    logging.info(f"Unsuccessful download - {exec_count}")


def make_list(url : str, count : int) -> list:
    """
    Make list of images tags

    Make list of images tags using url
    Parameters
    ----------
    url : str
        Путь для скачивания
    count : int
        Кол-во изображений для скачивания
    Returns
    ----------
    list
        Список с тегами картинок 
    """
    list_img = []
    url = url.replace("object", OBJECT)
    new_url = url[:-1]
    try:
        for page in range(1, count):
            if(len(list_img) >= count): return list_img
            new_page = f"{new_url}{page}"

            sleep_time = random.uniform(2, 10)
            time.sleep(sleep_time)
            html = requests.get(new_page, headers=HEADERS) 

            soup = BeautifulSoup(html.text, "lxml")
            list_img += soup.find_all("img", class_="mimg")
    except Exception as err:
        logging.error(f"{err}", exc_info=True)
    return list_img


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename=os.path.join("py_log.log"), filemode="w")

    with open(os.path.join("Lab1", "input_data.json"), 'r') as fjson:
        fj = json.load(fjson)

    URL = fj["search_url"]
    HEADERS = fj["headers"]
    COUNT = fj["counter"]
    MAIN_FOLDER =  fj["main_folder"]
    OBJECT = fj["object"]

    images = make_list(URL, COUNT)
    download(images, os.path.join(MAIN_FOLDER, OBJECT).replace("\\", "/"), COUNT)
