"""Module providing a function printing python version 3.11.5."""
import os
from time import sleep
import requests
from bs4 import BeautifulSoup
from fake_headers import Headers


def create_folders(name:str)->None:
    """This function create a folder"""
    if not os.path.exists("Lab1/Lab1-var6/dataset"):
        os.makedirs(f"Lab1/Lab1-var6/dataset/{name}")
    elif  not os.path.exists(f"Lab1/Lab1-var6/dataset/{name}"):
        os.mkdir(f"Lab1/Lab1-var6/dataset/{name}")

def save_image(url: str, filename: str) -> None:
    """This func downloads the image from the link"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(1024):
                    if chunk:
                        file.write(chunk)
    except (Exception, requests.exceptions.RequestException):
        print('Ошибка при загрузке: ', url, ':', str(Exception, requests.exceptions.RequestException))

def yandex_images_iarser(text : str) -> []:
    create_folders(text)
    i = 0
    main_url = f"https://yandex.ru/images/search?from=tabbar&text={text}"
    headers = Headers(headers=True).generate()
    result = requests.get(main_url, headers, timeout= 10)
    print(result)
    soup = BeautifulSoup(result.content, "lxml")
    links = soup.findAll("img",
                         class_ = "serp-item__thumb justifier__thumb"
                        )
    for link in links:
        try:
            link = link.get("src")
            save_image("http:" + link, f"Lab1/Lab1-var6/dataset/{text}/{i}.jpg")
            i += 1
            print(i)
            sleep(1)
        except Exception:
            continue

if __name__ == "__main__":
    yandex_images_iarser("tiger")
    yandex_images_iarser("leopard")
