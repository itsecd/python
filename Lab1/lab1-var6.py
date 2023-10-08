import os
import requests
import logging
import string
from bs4 import BeautifulSoup
from settings import (
    FOLDER_TIGER,
    FOLDER_LEOPARD,
    COUNT_FOR_LIST,
    COUNT_FOR_DOWNLOADS,
    SEARCH_TIGER,
    SEARCH_LEOPARD,
)
from typing import Dict, List

HEADERS = {
   "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
   "Referer":"https://www.bing.com/"
}
logging.basicConfig(filename="py_log.log", filemode="w", level=logging.DEBUG)



def create_folder(folder_name: str) -> str:
    try:
        if not os.path.exists(f"dataset/{folder_name}"):
            os.mkdir(f"dataset/{folder_name}")
    except Exception as e:
        logging.error(f"Error creating folder: {str(e)}")
    logging.info("folder created")


def create_list(url: str) -> list:
    list = []
    count = 0
    try:
        for pages in range(1, 30):
            if count >= int(COUNT_FOR_LIST):
                break
            url_pages: str = f"{url[:-1]}{pages}"
            response = requests.get(url_pages, headers=HEADERS)
            soup = BeautifulSoup(response.text, "lxml")
            images = soup.find_all("img")
            list+=images
            count += 1
        return list
    except Exception as e:
        logging.error(f"list not created: {str(e)}")
    logging.info("img uploaded to list")


def download_images(url: str, folder_name: str) -> str:
    list=create_list(url)
    logging.info("ready for download")
    num=0
    for img_tag in list:
        if len(list) < int(COUNT_FOR_DOWNLOADS):
            continue
        try:
            src = img_tag["src"]
            tag=img_tag.find('rp')
            if(tag!=-1):
                list.remove(img_tag)
                img_tag+=1
                break
            print(src)
            response = requests.get(src)
            numbers = format(num).zfill(4)
            create_folder(folder_name)
            try:
                with open(os.path.join("dataset", f"{folder_name}", f"{numbers}.jpg"), "wb") as f:
                    f.write(response.content)
                    num += 1
            except Exception as e:
                    logging.error(f"Error creating file: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing link:{str(e)}")
    logging.info("Images downloaded")


if __name__ == "__main__":
    os.mkdir("dataset")
    download_images(SEARCH_TIGER, FOLDER_TIGER)
    download_images(SEARCH_LEOPARD, FOLDER_LEOPARD)
