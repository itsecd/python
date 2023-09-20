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

def download_images(images, folder_name):
  count=0
  print(f"Total {len(images)} Image Found!")
  if len(images) != 0:
    for i, image in enumerate(images):
      try:
        image_link=image["src"]
      except:
        pass     
      try:
        r=requests.get(image_link).content
        try:
          r = str(r, 'utf-8')
        except UnicodeDecodeError:
          with open(f"{folder_name}/{i}.jpg", "wb+") as f:
            f.write(r)
          count+=1
      except:
        pass
    if count==len(images):
      print("All Images Downloaded!")
    else:
      print(f"Not all of {count} images are Downloaded")

def main(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    images = soup.findAll('img')
    folder_create(images)
os.mkdir('dataset')
url = input("Enter site URL:")
main(url)
          

         