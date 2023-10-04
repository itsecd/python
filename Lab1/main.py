import os
import os.path
import requests
from bs4 import BeautifulSoup

HEADERS={"User-Agent": "Mozilla/5.0"}
MAIN_FOLDER="dataset"
FOLDER_CAT="cat"
FOLDER_DOG="dog"
url_cat="https://yandex.ru/images/search?text=cat.jpg"
url_dog="https://yandex.ru/images/search?text=dog.jpg"

def create_directory(folder: str)->str:
    try:
        if not os.path.exists(folder): #возвращает false, если путь не существует
            os.makedirs(folder) #создает промежуточные каталоги по пути folder, если они не существуют
    except: 
        print("Folder don't create")

def make_lists(url:str)->list:
    list_url=[]
    html = requests.get(url_cat, HEADERS)
    soup = BeautifulSoup(html.text, "lxml")
    print("List don't create") 
 