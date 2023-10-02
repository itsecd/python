import os
import requests
import logging
from bs4 import BeautifulSoup
from typing import Dict, List

#Directing the log entry to a file
logging.basicConfig(filename = "py_log.log",filemode='w', level=logging.DEBUG)

"""
function download_images Downloads images from list
"""
def download_images(my_list:list,folder_name:str):
    try:
     os.mkdir(f'dataset\{folder_name}')
     num = 0
     logging.info('Uploading images')
     for i in my_list:
        if num!=1000:
         response = requests.get(i)
         with open(os.path.join("dataset",f"\{folder_name}",f"{format(num).zfill(4)}.jpg"), 'wb') as f:
            f.write(response.content)
         num += 1
        else:
           break
     logging.info(f'Images uploaded:{num}')
    except Exception as e:  
        logging.error('Error in download_images: ' + str(e))

"""
function correct_images checks for duplicate links and removes them
Returns a sorted list
"""

def correct_images(my_list:list,folder_name:str)->list:
    sorted_list = list(set(my_list))
    logging.info('the list is sorted')
    download_images(sorted_list,folder_name)

"""
The function searches for all src references and adds them to the list.
Returns a list
"""

def create_list(search:str,folder_name:str)->list:
    logging.info('Create list')
    list = []
    count = 0
    for pages in range(1,33):
        url = f"https://www.yandex.ru/images/search?lr=51&p={pages}&rpt=image&{search}"
        response = requests.get(url).text
        soup = BeautifulSoup(response, 'lxml')
        images = soup.find_all('img')
        for image in images[1:]:
            if count==1200:
             break
            count += 1
            list.append("https:"+image['src'])
    logging.info("list created,ready for sorted")
    correct_images(list,folder_name)

"""
Create a dataset folder in which we will store 
folders with images and pass the url
"""
os.mkdir('dataset')
create_list(os.getenv("TEXT_TIGER"),os.getenv("FOLDER_TIGER"))
create_list(os.getenv("TEXT_LEOPARD"),os.getenv("FOLDER_LEOPARD"))