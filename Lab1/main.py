import os
import requests
from bs4 import BeautifulSoup

#Создание папки в случае отсутствия
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
