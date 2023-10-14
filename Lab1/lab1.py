import os
import requests
from bs4 import BeautifulSoup


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_images(query, num_images):
    create_directory("dataset")
    count = 0
    page = 1
    while count < num_images:
        search_url = f"https://www.bing.com/images/search?q={query}&form=HDRSC2&first={page}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "lxml")
        img_tags = soup.find_all('img', {"src": True}, class_='mimg')

        for img_tag in img_tags:
            img_url = img_tag["src"]
            if img_url:
                try:
                    img_data = requests.get(img_url, headers=headers, timeout=10).content
                    class_folder = f"dataset/{query}"
                    create_directory(class_folder)
                    with open(f"{class_folder}/{count:04}.jpg", "wb") as img_file:
                        img_file.write(img_data)
                    count += 1
                    if count >= num_images:
                        break
                except Exception as e:
                    print(f"Error downloading image")
        page += 1


if __name__ == "__main__":
    classes = ["polar bear", "brown bear"]
    num_images_per_class = 1000
    for class_name in classes:
        download_images(class_name, num_images_per_class)