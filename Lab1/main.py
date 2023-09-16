# libaries
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup
import requests
import re
from re import sub
from decimal import Decimal
import io
from datetime import datetime

#запросы и получение html кода страницы
URL_ROSE="https://yandex.ru/images/search?text=rose.jpg"

HTML_ROSE=requests.get(URL_ROSE).text

soup_rose=BeautifulSoup(HTML_ROSE, 'lxml')
#r=soup_rose.find('div', class_='serp-item__preview').find('a', class_='serp-item__link').get('href')