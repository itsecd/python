import os
import requests
from bs4 import BeautifulSoup


#Создание папки в случае отсутствия
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_html_page(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    return response.text

def download_img(search_query, save_directory, num_images=1000):
    create_directory(save_directory)

    for start in range(0, num_images, 100):
        
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

    print(f"Загружено {num_images} изображений класса {search_query}")

classes = ["polar bear", "brown bear"]
num_images_for_class = 1000

for class_name in classes:
    save_directory = os.path.join("dataset", class_name)
    download_img(class_name, save_directory, num_images=num_images_for_class)