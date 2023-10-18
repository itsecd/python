from bs4 import BeautifulSoup
import requests
import os
import re
import codecs
import chardet
import logging


def get_page(URL: str) -> str:
    html_page = requests.get(URL, headers={"User-Agent":"Mozilla/5.0"})
    encode = chardet.detect(html_page.content)['encoding']
    decoded_html_page = html_page.content.decode(encode)
    soup = BeautifulSoup(decoded_html_page, features="html.parser")
    return soup

def create_dataset_folder(name:str) -> None:
    if not os.path.exists(f"{name}"):
        os.mkdir(name)
        for i in range(0, 6):
            os.mkdir(f"{name}/{i}")

