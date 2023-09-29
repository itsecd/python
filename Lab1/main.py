# libaries
import os
import shutil
from bs4 import BeautifulSoup
import requests

# функции для создания папок, скачивания изображений и удаления папок


def create_directory(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)


def download(url, folder):
    count = 0
    for link in url:
        if count > 1010:
            break
        try:
            src = link["src"]
            response = requests.get(src)
            with open(f"{folder}/{count:04}.jpg", "wb") as file:
                file.write(response.content)
                count += 1
        except:
            "Uncorrect URL"


def delete(folder: str):
    shutil.rmtree(folder)


# получаем все ссылки на розы/тюльпаны списком
list_url_rose = []
for pages in range(1, 120):
    url_rose = f"https://www.bing.com/images/search?q=rose.jpg&redig=7D4B2E55AA5E4A4CA223FB76FBC6D258&first={pages}"
    html_rose = requests.get(url_rose, headers={"User-Agent": "Mozilla/5.0"})
    soup_rose = BeautifulSoup(html_rose.text, "lxml")
    roses = soup_rose.findAll("img")
    list_url_rose += roses


list_url_tulip = []
for pages in range(1, 120):
    url_tulip = f"https://www.bing.com/images/search?q=tulip.jpg&qs=UT&form=QBIR&sp=1&lq=0&pq=tulip.jpg&sc=2-9&cvid=9577F520591A403C88168B8637C22677&first={pages}"
    html_tulip = requests.get(url_tulip, headers={"User-Agent": "Mozilla/5.0"})
    soup_tulip = BeautifulSoup(html_tulip.text, "lxml")
    tulip = soup_tulip.findAll("img")
    list_url_tulip += tulip


# создаем папку и скачиваем все файлы из полученных списков
len_folder_roses = 0
for rose in list_url_rose:
    if len_folder_roses > 1010:
        break
    else:
        len_folder_roses += 1
        create_directory("dataset/rose")
        download(list_url_rose, "dataset/rose")
    break


len_folder_tulips = 0
for tulip in list_url_tulip:
    if len_folder_tulips > 1010:
        break
    else:
        len_folder_tulips += 1
        create_directory("dataset/tulip")
        download(list_url_tulip, "dataset/tulip")
    break
