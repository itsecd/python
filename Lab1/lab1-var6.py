import os
import requests
from bs4 import BeautifulSoup
from typing import Dict, List
"""
function download_images Downloads images from list
"""
def download_images(my_list:list)->int:
    try:
     folder_name=os.mkdir(f'dataset/{os.getenv("FOLDER_TIGER")}')
     num = 0
     for i in my_list:
        while num!=300:
         response = requests.get(i)
         with open(f"dataset/{folder_name}/{os.path.join(format(num).zfill(4))}.jpg", 'wb') as f:
            f.write(response.content)
         num += 1
    except: 
       print('No file by that name')

"""
function correct_images checks for duplicate links and removes them
Returns a sorted list
"""

def correct_images(my_list:list)->list:
    print('total: ', len(my_list))
    sorted_list = list(set(my_list))
    print('total: ', len(sorted_list))
    download_images(sorted_list)

"""
The function searches for all src references and adds them to the list.
Returns a list
"""

def create_list(search:str)->list:
    list = []
    count = 0
    for pages in range(1,10):
        for i in search:
            while i!=pages
        response = requests.get(url).text
        soup = BeautifulSoup(response, 'lxml')
        images = soup.find_all('img')
        for image in images[1:]:
            while count!=300:
             count += 1
             list.append("https:"+image['src'])
    correct_images(list)

"""
Create a dataset folder in which we will store 
folders with images and pass the url
"""
os.mkdir('dataset')
create_list(os.getenv("SEARCH_TIGER"))
create_list(os.getenv("SEARCH_LEOPARD"))