import os
import requests 
import cv2
from bs4 import BeautifulSoup

def create_folder(folder_name):
    if not os.path.exist(folder_name):
        os.makedirs(folder_name)

def dowload_image(url,folder_name,image_number):
    response=requests.get(url)
    file_name=f"{folder_name}/{str(image_number).zfill(4)}.jpg"
    with open(file_name,"wb") as file:
        file.write(response.content)