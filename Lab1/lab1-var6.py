import os
import requests
import cv2
from bs4 import BeautifulSoup
"""
def folder_create():
 try:
    folder_name=input("Enter Folder name:- ")
    os.mkdir(f'dataset/{folder_name}')
 except:
    print("Folder Exist with that name!")
    folder_create()
"""
def download_images(url, folder_name,split):
  r = requests.get(url,headers={"User-Agent":"Mozilla/5.0"})
  soup = BeautifulSoup(r.text, 'lxml')
  for p in range(1,6):
    urls=f"https://www.yandex.ru/images/search?lr=51&p={p}&rpt=image&{split}"
    images = soup.find_all('img')
  count=0
  print(f"Total {len(images)} Image Found!")
  if len(images) != 0:
    count=0
    for image in images[1:]:
        src="https:"+image['src']
        response=requests.get(src, allow_redirects=True)
        with open(f"dataset/{folder_name}/{count}.jpg",'wb') as f:
           f.write(response.content)
           count+=1
    if (count==len(images)):
       print("All Images are Downloaded!")

    else:
       print("Houston, we have a problem")   


os.mkdir('dataset')
url = input("Enter site URL:")
split=(url.split('?'))[1]
folder_name=input("Enter Folder name:- ")
os.mkdir(f'dataset/{folder_name}')
download_images(url,folder_name,split)
          
#URL for tigers: https://yandex.ru/images/search?text=tiger
#URL for leopards: https://yandex.ru/images/search?text=leopard
         