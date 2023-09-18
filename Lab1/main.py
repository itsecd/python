# libaries
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup
import requests
import re
import os
from re import sub
from decimal import Decimal
import io
from datetime import datetime


def create_directory(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)




# запросы и получение html кода страницы
URL_ROSE = "https://yandex.ru/images/search?text=rose.jpg" 
URL_TULIP = "https://yandex.ru/images/search?from=tabbar&text=tulip.jpg"

HTML_ROSE = requests.get(URL_ROSE).text
HTML_TULIP = requests.get(URL_TULIP).text

SOUP_ROSE = BeautifulSoup(HTML_ROSE, 'lxml')
SOUP_TULIP = BeautifulSoup(HTML_TULIP, 'lxml')



r = SOUP_ROSE.find('div', class_='serp-item__preview')  #.find('a', class_='serp-item__link').get('href')
t = SOUP_ROSE.find('div', class_='serp-item__preview') #.find('a', class_='serp-item__link').get('href')

print(r)
print(t)
create_directory("dataset")
create_directory("dataset/rose")
create_directory("dataset/tulip")








# создание папки с помощью библиотеки os

