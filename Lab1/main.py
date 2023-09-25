import os
import shutil
import requests
from bs4 import BeautifulSoup

def delete(folder:str):
    shutil.rmtree(folder) 
# удаляет текущую директорию и все поддиректории 

URL_CAT = "https://yandex.ru/"
URL_DOG = "https://yandex.ru/"

HTML_CAT = requests.get(URL_CAT, headers={"User-Agent":"Mozilla/5.0"})
HTML_DOG = requests.get(URL_DOG, headers={"User-Agent":"Mozilla/5.0"})

SOUP_CAT = BeautifulSoup(HTML_CAT.text, 'lxml')
SOUP_DOG = BeautifulSoup(HTML_DOG.text, 'lxml')
