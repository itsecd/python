from bs4 import BeautifulSoup
import requests
import os

url_rose = "https://yandex.ru/images/search?lr=51&text=rose.jpg"
url_tulip = "https://yandex.ru/images/search?lr=51&text=tulip.jpg"

headers = {
    "Accept" : "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
}

html_rose = requests.get(url_rose, headers=headers)
html_tulip = requests.get(url_tulip, headers=headers)

soup_rose = BeautifulSoup(html_rose.text, "lxml") 
print(soup_rose)
