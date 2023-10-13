import os
import requests 
import cv2
from bs4 import BeautifulSoup

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def dowload_image(image_urls,folder_name):
    for i, url in enumerate(image_urls):
        response=requests.get(url)
        if response.status_code==200:
            file_name=f"dataset/{folder_name}/{i:04d}.jpg"
        with open(file_name,"wb") as file:
            file.write(response.content)

def get_images_urls(query, count): 
    urls=[]
    page=1
    while len(urls)<count:
        url=f"https://www.bing.com/images/search?q={query}&form=hdrsc2&first={page}"
        response=requests.get(url) 
        soup=BeautifulSoup(response.text,"html.parser")
        images=soup.find_all("img")
        for image in images:
            image_url=image.get("src")
            if image_url and image_url.startswith("https://"):
                urls.append(image_url)
        page+=1
    return urls[:count]

queries=["tiger","leopard"]
create_folder("dataset")
for query in queries:
    folder_name=f"dataset/{query}"
    create_folder(folder_name)
tiger_urls=get_images_urls("tiger",50)  
dowload_image(tiger_urls,"tiger")
leopard_urls=get_images_urls("leopard",50)
dowload_image(leopard_urls,"leopard")