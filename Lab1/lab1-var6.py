import os
import requests
import cv2
from bs4 import BeautifulSoup


def main(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    images = soup.findAll('img')
os.mkdir('dataset')
url = input("Enter site URL:")
main(url)
          

         