from bs4 import BeautifulSoup
import requests
import os

COUNT = 2000

def create_folder(folder : str):
    if not os.path.isdir(folder): 
        os.makedirs(folder)

def download(urls, folder : str):
    count = 0
    for flower_url in urls:
        try:
            src=flower_url['src']
            response = requests.get(src)
            with open(f"{folder}/{count:04}.jpg", "wb") as file:
                file.write(response.content)
                count+=1
        except: "Uncorect URL" 
        if(count >= COUNT):
            break
    

_headers = {
   "User-Agent":"Mozilla/5.0"
}

roses_list_url = []
tulips_list_url = []


for page in range(1, 999):
    url_rose = ("https://www.bing.com/images/search?q=rose.jpg&go=%D0%9F%D0%B"+ 
    "E%D0%B8%D1%81%D0%BA&qs=ds&form=QBIR&first={page}")
    url_tulip = ("https://www.bing.com/images/search?q=tulip.jpg&qs=n&form=QB"+ 
    "IR&sp=-1&lq=0&pq=tulip.jp&sc=6-8&cvid=973A3A4B92834C4B8848B29F7D753069&g"+ 
    "hsh=0&ghacc=0&first={page}")

    html_rose = requests.get(url_rose, headers=_headers) # Get a html code
    html_tulip = requests.get(url_tulip, headers=_headers) 

    soup_rose = BeautifulSoup(html_rose.text, "lxml") # Get a BS object
    soup_tulip = BeautifulSoup(html_tulip.text, "lxml") 

    roses_list_url += soup_rose.find_all("img", class_="mimg") # Get a list of <img> 
    tulips_list_url += soup_tulip.find_all("img", class_="mimg")

    if(len(roses_list_url) >= COUNT and len(tulips_list_url) >= COUNT):
        break

create_folder("Lab1/dataset/roses") # Create folders
create_folder("Lab1/dataset/tulip")

download(roses_list_url, "Lab1/dataset/roses") # Download images
download(tulips_list_url, "Lab1/dataset/tulip")

print(len(roses_list_url))
print(len(tulips_list_url))
