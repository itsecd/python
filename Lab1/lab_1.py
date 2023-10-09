import os
import requests
from bs4 import BeautifulSoup
import hashlib

# Функция для создания директории, если она не существует
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Функция для получения хэша изображения
def get_image_hash(image_data):
    md5 = hashlib.md5()
    md5.update(image_data)
    return md5.hexdigest()