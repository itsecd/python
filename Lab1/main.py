# libaries
import os
from bs4 import BeautifulSoup
import requests
import cv2



def create_directory(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)

def download(url):
    try: 
        response=requests.get(url=url)
        for i in range(1000):
         with open(f'dataset/rose/{i}.jpg', 'wb') as file:
            file.write(response.content)
    except: 'Uncorrect URL'

#def delete(folder:str):
        os.remove(folder) 
   

# запросы и получение html кода страницы
URL_ROSE = "https://yandex.ru/images/search?text=rose.jpg" 
URL_TULIP = "https://yandex.ru/images/search?from=tabbar&text=tulip.jpg"

HTML_ROSE = requests.get(URL_ROSE, headers={"User-Agent":"Mozilla/5.0"})
HTML_TULIP = requests.get(URL_TULIP, headers={"User-Agent":"Mozilla/5.0"})

SOUP_ROSE = BeautifulSoup(HTML_ROSE.text, 'lxml')
SOUP_TULIP = BeautifulSoup(HTML_TULIP.text, 'lxml')



r = SOUP_ROSE.find_all('img')
t = SOUP_ROSE.find_all('img')
URL='https://avatars.mds.yandex.net/i?id=17291e9f90aafdd133916ce64c0f3be9ccd19850-10311215-images-thumbs&n=13'


print(r)
create_directory("dataset")
create_directory("dataset/rose")
create_directory("dataset/tulip")
#download(url=URL)
#cv2.imwrite('dataset/rose', 'i.jpg' )

delete("dataset/rose")




# создание папки с помощью библиотеки os

