import os
import requests 
import cv2
from bs4 import BeautifulSoup

def get_images_urls(query, count): 
    urls=[]
    page=1
    while len(urls)<count:
        url=f"https://yandex.ru/images/search?lr=51&p={query}&itype=jpg&p={page}"
        response=requests.get(url) 
        soup=BeautifulSoup(response.text,"html.parser")
        images=soup.find_all("img")
        for image in images:
            image_url=image.get("src")
            if image_url and image_url.startswith("https://"):
                urls.append(image_url)
        page+=1
    return urls[:count]
tiger_urls=get_images_urls("tiger",10)  
print(tiger_urls)   