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

        img_tags = soup.find_all("img", class_="serp-item__thumb")
        for i, img_tag in enumerate(img_tags):
            img_url = img_tag.get("src")
            img_data = requests.get(img_url).content
            img_filename = f"{i+start:04d}.jpg"
            img_path = os.path.join(save_directory,img_filename)
            with open(img_path, "wb") as img_file:
                img_file.write(img_data)

            print(f"Скачано изображение {img_filename}")

    print(f"Загружено {num_images} изображений")

classes = ["polar bear", "brown bear"]
num_images_for_class = 1000

for class_name in classes:
    save_directory = os.path.join("dataset", class_name)
    download_img(class_name, save_directory, num_images=num_images_for_class)