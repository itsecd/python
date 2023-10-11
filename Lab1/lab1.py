import os
import requests 
import cv2
from bs4 import BeautifulSoup

def create_folder(folder_name):
    if not os.path.exist(folder_name):
        os.makedirs(folder_name)

