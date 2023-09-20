import os
import requests
from bs4 import BeautifulSoup


#Создание папки в случае отсутствия
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_img(search_query, save_directory, num_images=1000):
    create_directory(save_directory)
    base_url = "https://yandex.ru/images/"

    for start in range(0, num_images, 10):
        params = {
            "text": search_query,
            "type": "photo",
            "from": "tabbar",
            "p": str(start),
        }

        response = requests.get(base_url,params=params)
        soup = BeautifulSoup(response.text, "html.parser")

        