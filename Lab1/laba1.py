from bs4 import BeautifulSoup
import requests
import os

def create_folder(folder : str):
    if not os.path.isdir(folder): 
        os.mkdir(folder)

def download(src, folder : str):
    count = 0
    for rose in src:
        src = rose["src"]
        response = requests.get(src)
        with open(f"{folder}/{count}.jpg", "wb") as file:
            file.write(response.content)
            count+=1
    


url_rose = "https://yandex.ru/images/search?lr=51&text=rose.jpg"
url_tulip = "https://yandex.ru/images/search?lr=51&text=tulip.jpg"

headers = {
    "Accept" : "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
}

html_rose = requests.get(url_rose, headers=headers)
html_tulip = requests.get(url_tulip, headers=headers)

soup_rose = BeautifulSoup(html_rose.text, "lxml") 
soup_tulip = BeautifulSoup(html_tulip.text, "lxml") 

roses_list = soup_rose.find_all("img", class_="serp-item__thumb justifier__thumb")
tulips_list = soup_tulip.find_all("img", class_="serp-item__thumb justifier__thumb")

create_folder("Lab1/dataset/roses")
create_folder("Lab1/dataset/tulip")

download(roses_list, "Lab1/dataset/roses")
download(tulips_list, "Lab1/dataset/tulip")
# print(roses_list)