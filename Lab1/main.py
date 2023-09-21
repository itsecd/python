# libaries
import os
import shutil
from bs4 import BeautifulSoup
import requests



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
        if count+2>len(url):
                break

def delete(folder:str):
    shutil.rmtree(folder) 
       

URL_ROSE = "https://www.google.com/search?sca_esv=566872717&q=rose.jpg&tbm=isch&source=lnms&sa=X&ved=2ahUKEwjk0ond4LiBAxUyExAIHadUCPMQ0pQJegQIEBAB&biw=1036&bih=810&dpr=1.25" 
URL_TULIP = "https://www.google.com/search?q=tulip.jpg&tbm=isch&ved=2ahUKEwi71JmWmbmBAxWTJxAIHaCXBiMQ2-cCegQIABAA&oq=tulip.jpg&gs_lcp=CgNpbWcQARgAMgcIABATEIAEMggIABAHEB4QEzIICAAQBxAeEBMyCAgAEAcQHhATOgUIABCABDoGCAAQBxAeUKcXWIlMYOlcaARwAHgAgAH0BIgBigySAQYxMS41LTGYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=ceYKZfvXEpPPwPAPoK-amAI&bih=810&biw=1036"

HTML_ROSE = requests.get(URL_ROSE, headers={"User-Agent":"Mozilla/5.0"})
HTML_TULIP = requests.get(URL_TULIP, headers={"User-Agent":"Mozilla/5.0"})


SOUP_ROSE = BeautifulSoup(HTML_ROSE.text, 'lxml')
SOUP_TULIP = BeautifulSoup(HTML_TULIP.text, 'lxml')

roses=SOUP_ROSE.findAll('img')
tulips=SOUP_TULIP.findAll('img')

for rose in roses:
    len_folder_roses = len(os.listdir(path="dataset/rose"))
    len_roses=len(roses)
    if len_folder_roses+2>len_roses:
         break
    else:
        create_directory("dataset/rose")
        download(roses, "dataset/rose")
       
for tulip in tulips:
    len_folder_tulips = len(os.listdir(path="dataset/tulip"))
    len_tulip=len(tulips)
    if len_folder_tulips+2>len_tulip:
         break
    else:
        create_directory("dataset/tulip")
        download(tulips, "dataset/tulip")




