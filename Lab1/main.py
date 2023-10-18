import os
from bs4 import BeautifulSoup
import requests
import chardet
import logging
import argparse

BASE_URL = "https://otzovik.com/reviews/sberbank_rossii/1/?ratio=1" #Пока заглушка
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

logging.basicConfig(level=logging.INFO)

def create_folder(name:str) -> None: #Создает папки в корневой папке лабы, норм?
    try:
        if not os.path.exists(f"{name}"):
            os.mkdir(name)
            for i in range(1, 6):
                os.mkdir(f"{name}/{i}")
    except Exception as exc:
        logging.exception(f"Can not create folder: {exc.message}\n{exc.args}\n")

def get_page(URL: str) -> str: #Работает нормально, стоит переписать, возможно
    html_page = requests.get(URL, headers=HEADERS, timeout=5)
    encode = chardet.detect(html_page.content)['encoding']
    decoded_html_page = html_page.content.decode(encode)
    soup = BeautifulSoup(decoded_html_page, features="html.parser")
    return soup

def w_review_to_txt_file(soup: str, dataset_name: str)-> dict[str, int]:
    



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Input directory path, link for parsing, count of ___")
    parser.add_argument("-p", "--path", help="Input directory path", type=str)
    parser.add_argument("-l", "--link", help="Input link", type=str)
    parser.add_argument("-c", "--count", help="input count of ___")
    args = parser.parse_args()
    w_review_to_txt_file()