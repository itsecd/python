import os
import requests
import cv2
from bs4 import BeautifulSoup

def folder_create(images):
 try:
    folder_name=input("Enter Folder name:- ")
    os.mkdir(f'dataset/{folder_name}')
 except:
    print("Folder Exist with that name!")
    folder_create()
 download_images(images,f'dataset/{folder_name}')


def main(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    images = soup.findAll('img')
    folder_create(images)
os.mkdir('dataset')
url = input("Enter site URL:")
main(url)
          

         