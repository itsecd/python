# libaries
import os
from bs4 import BeautifulSoup
import requests
import cv2
import shutil



def create_directory(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)

def download(url, folder):
    count=0
    for link in url:
        try: 
            src=link['src']
            response=requests.get(src)
            with open(f'{folder}/{count}.jpg', 'wb') as file:
                file.write(response.content)
                count+=1
        except: 'Uncorrect URL'

def delete(folder:str):
    shutil.rmtree(folder) 
       

URL_ROSE = "https://www.google.com/search?sca_esv=566872717&q=rose.jpg&tbm=isch&source=lnms&sa=X&ved=2ahUKEwjk0ond4LiBAxUyExAIHadUCPMQ0pQJegQIEBAB&biw=1036&bih=810&dpr=1.25" 
URL_TULIP = "https://www.bing.com/images/search?q=tulip.jpg&form=HDRSC2&first=1"

HTML_ROSE = requests.get(URL_ROSE, headers={"User-Agent":"Mozilla/5.0"})
HTML_TULIP = requests.get(URL_TULIP, headers={"User-Agent":"Mozilla/5.0"})

SOUP_ROSE = BeautifulSoup(HTML_ROSE.text, 'lxml')
SOUP_TULIP = BeautifulSoup(HTML_TULIP.text, 'lxml')

t = SOUP_ROSE.find_all('img')

data=[]
roses=SOUP_ROSE.findAll('img', class_='DS1iW')
#link=SOUP_ROSE.find('img', class_='DS1iW').get('src')
links=SOUP_ROSE.findAll('img')

for link in links:
     create_directory("dataset/rose")
     download(links, "dataset/rose")





