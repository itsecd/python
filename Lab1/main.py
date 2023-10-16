import os
import logging
import requests
from bs4 import BeautifulSoup


"""The function creates a folder if it does not exist"""
def create_directory(directory: str) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as exc:
        logging.exception(f"Can't create folder: {exc.message}\n{exc.args}\n")

def download_img(search_query, save_directory, num_images=1000):
    create_directory(save_directory)

    for start in range(1, num_images, 1):
        
        base_url = f"https://www.bing.com/images/search?q={search_query}&form=HDRSC2&first={start}"
        try:
            response_text = get_html_page(base_url)  # Получаем HTML-страницу
            soup = BeautifulSoup(response_text, "html.parser")

            img_tags = soup.find_all("img", class_="mimg")
            for i, img_tag in enumerate(img_tags):
                img_url = img_tag.get("src")
                img_data = requests.get(img_url).content
                img_filename = f"{i+start:04d}.jpg"
                img_path = os.path.join(save_directory,img_filename)

                with open(img_path, "wb") as img_file:
                    img_file.write(img_data)

                print(f"Скачано изображение {img_filename}")
        except requests.exceptions.MissingSchema:
         print(f"Проблема с URL: {base_url}")
           # continue

    print(f"Загружено {num_images} изображений класса {search_query}")

classes = ["polar_bear", "brown_bear"]
num_images_for_class = 1000

for class_name in classes:
    save_directory = os.path.join("dataset", class_name)
    download_img(class_name, save_directory, num_images=num_images_for_class)