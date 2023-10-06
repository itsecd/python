import os
import logging
import requests
import argparse
from bs4 import BeautifulSoup


parser = argparse.ArgumentParser(description = "Count and type of images")

MAIN_FOLDER = "Lab1"
FOLDER = "dataset"
HEADERS = {
   "User-Agent":"Mozilla/5.0"
}

logging.basicConfig(level=logging.INFO, filename=os.path.join(MAIN_FOLDER, "py_log.log"), filemode="w")


def create_folder(folder: str):
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


def download(list_images : list, folder : str):
    """
    Download images

    Download images from list
    Parameters
    ----------
    list_images : list
        Список тегов с картинками
    folder : str
        Путь для скачивания
    """
    count = 0
    exec_count = 0
    for flower_url in list_images:
        if count > args.count : break
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
    new_url = url[:-1]
    pages = int(count/10)
    try:
        for page in range(1, pages):
            new_page = f"{new_url}{page}"
            html = requests.get(new_page, headers=HEADERS) 
            soup = BeautifulSoup(html.text, "lxml")
            list_img += soup.find_all("img", class_="mimg")
        return list_img
    except Exception as err:
        logging.error(f"{err}", exc_info=True)


if __name__ == "__main__":

    parser.add_argument('-c','--count',type=int, default=1000, help='Count of images')
    parser.add_argument('-o', '--object', type=str, default='rose', help='Type of object for search')
    args = parser.parse_args()

    url_images= f"https://www.bing.com/images/search?q={args.object}.jpg&first=1"
    count_images = args.count

    images = make_list(url_images, count_images)
    download(images, os.path.join(MAIN_FOLDER, FOLDER, args.object).replace("\\", "/"))
