"""Module providing a function printing python version 3.11.5."""
import os
from time import sleep
import requests
from bs4 import BeautifulSoup


def create_folders(name:str) -> None:
    """This function create a folder"""
    if not os.path.exists("dataset"):
        os.makedirs(os.path.join("dataset", name))
    elif  not os.path.exists(os.path.join("dataset", name)):
        os.mkdir(os.path.join("dataset", name))

def make_path_and_filename(index: int, path: str) -> str:
    """Сreates the path to the future file and its name"""
    filename = f'{index:04d}' + ".jpg"
    return os.path.join("dataset", path, filename)

def save_image(url: str, filename: str) -> None:
    """This func downloads the image from the link"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.ok:
            with open(filename, 'wb') as file:
                file.write(response.content)
    except (Exception, requests.exceptions.RequestException):
        print('Unable to download image: ',
              url,
              ':',
              str(Exception, requests.exceptions.RequestException)
              )

def yandex_images_iarser(text : str) -> []:
    """parser 'Yandex.Images'"""
    create_folders(text)
    i = 0
    for page in range(20):
        main_url = f"https://yandex.ru/images/search?from=tabbar&text={text}&p={page}"
        headers = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"\
            "Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.47"
        result = requests.get(main_url, headers, timeout= 10)
        print(result)
        soup = BeautifulSoup(result.content, features = "lxml")
        links = soup.findAll("img",
                            class_ = "serp-item__thumb justifier__thumb"
                            )
        for link in links:
            try:
                link = link.get("src")
                path_to_file = make_path_and_filename(i, text)
                if os.path.exists(path_to_file):
                    continue
                save_image("http:" + link,
                        path_to_file
                        )
                i += 1
                print(i)
                sleep(3)
            except Exception:
                print("Error")
                continue

if __name__ == "__main__":
    yandex_images_iarser("tiger")
    yandex_images_iarser("leopard")
