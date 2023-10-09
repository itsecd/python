import os
import requests
from bs4 import BeautifulSoup
import hashlib

# Функция для создания директории, если она не существует
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Функция для получения хэша изображения
def get_image_hash(image_data):
    md5 = hashlib.md5()
    md5.update(image_data)
    return md5.hexdigest()

# Функция для загрузки изображений по запросу
def download_images(query, num_images, num_pages=100):
    create_directory("dataset")
    count = 0
    downloaded_image_hashes = set()

    for page in range(1, num_pages + 1):
        if count >= num_images:
            break

        search_url = f"https://www.bing.com/images/search?q={query}&form=HDRSC2&first={page}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "lxml")
        img_tags = soup.find_all('img', {"src": True}, class_='mimg')

        for img_tag in img_tags:
            img_url = img_tag["src"]
            if img_url:
                try:
                    img_data = requests.get(img_url, headers=headers, timeout=10).content
                    img_hash = get_image_hash(img_data)

                    if img_hash not in downloaded_image_hashes:
                        class_folder = f"dataset/{query}"
                        create_directory(class_folder)
                        with open(f"{class_folder}/{count:04}.jpg", "wb") as img_file:
                            img_file.write(img_data)
                        downloaded_image_hashes.add(img_hash)
                        count += 1
                        if count >= num_images:
                            break
                except Exception as e:
                    print(f"Error downloading image")