import os
import requests
import logging
from bs4 import BeautifulSoup
from typing import Dict, List


logging.basicConfig(filename = "py_log.log",filemode='w', level=logging.DEBUG)


def download_images(my_list:list,folder_name:str)->int:
    """Downloads images from list"""
    try:
     os.mkdir(f'dataset/{folder_name}')
     num = 0
     logging.info('Uploading images')
     for i in my_list:
        if num!=os.getenv("COUNT_FOR_DOWNLOADS"):
         response = requests.get(i)
         numbers=format(num).zfill(4)
         with open(os.path.join("dataset",f"{folder_name}",f"{numbers}.jpg").replace("\\","/"), 'wb') as f:
            f.write(response.content)
         num += 1
        else:
           break
     logging.info(f'Images uploaded:{num}')
    except Exception as e:  
        logging.error('Error in download_images: ' + str(e))


"""
def correct_images(my_list:list,folder_name:str)->list:
    #function correct_images checks for duplicate links and removes them
    #Returns a sorted list

    sorted_list = list(set(my_list))
    logging.info('the list is sorted')
    download_images(sorted_list,folder_name)
"""


def create_list(search:str,folder_name:str)->list:
    """
    The function searches for all src references and adds them to the list.
    Returns a list
    """
    logging.info('Create list')
    list = []
    count = 0
    for pages in range(1,200):
        url = f"https://www.bing.com/images/search?q={search}&go=%D0%9F%D0%BE%D0%B8%D1%81%D0%BA&qs=ds&form=QBIRMH&first={pages}"
        response = requests.get(url).text
        soup = BeautifulSoup(response, 'lxml')
        images = soup.find_all('img', class_="mimg")
        for image in images[:1]:
           if count==os.getenv("COUNT_FOR_LIST"):
             break
           count += 1
           list.append(image['src'])
           print(list)
    logging.info("list created,ready for sorted")
    download_images(list,folder_name)

"""
Create a dataset folder in which we will store 
folders with images and pass the url
"""
os.mkdir('dataset')
create_list(os.getenv("TEXT_TIGER"),os.getenv("FOLDER_TIGER"))
create_list(os.getenv("TEXT_LEOPARD"),os.getenv("FOLDER_LEOPARD"))