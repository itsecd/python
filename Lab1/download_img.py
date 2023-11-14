import os
from site import abs_paths
import time
from tkinter import image_names
from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib.request
import csv

def make_directory(directory_cat, directory_dog):
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    if not os.path.exists(directory_cat):
        os.mkdir('dataset/cat')
    if not os.path.exists(directory_dog):
        os.mkdir('dataset/dog')


def scroll_driver(driver, height):
    scroll_range = 0
    while scroll_range < height:
        driver.execute_script(f"window.scrollTo(0, {scroll_range});") 
        scroll_range += 10
    

list_cat=[]
list_dog=[]

def make_driver(link, path_name, name):
    global list_cat,  list_dog
    driver = webdriver.Chrome()
    driver.get(link)
    time.sleep(4)
    scroll_driver(driver, 20000)
    list_pictures = driver.find_elements(By.XPATH, path_name)
    if name=="cat":
        list_cat += list_pictures
    if name=="dog":
        list_dog += list_pictures
    print(len(list_cat),len(list_dog))


def make_name(value):
    return '0'*(4-len(str(value))) + str(value)

def save_image():
    global list_cat,  list_dog
    directory_cat = "dataset/cat"
    directory_dog = "dataset/dog"
    make_directory(directory_cat, directory_dog)
    print(f'lens: {len(list_cat)}, {len(list_dog)}')

    for elem in range(len(list_cat)):
        img = urllib.request.urlopen(list_cat[elem].get_attribute('src')).read()
        out = open(f"{directory_cat}/{make_name(elem)}.jpg", "wb")
        out.write(img)
        out.close
    for elem in range(len(list_dog)):
        img = urllib.request.urlopen(list_dog[elem].get_attribute('src')).read()
        out = open(f"{directory_dog}/{make_name(elem)}.jpg", "wb")
        out.write(img)
        out.close

def main():
    for i in range(6):
        if len(list_cat) < 1000:
            make_driver(f"https://yandex.ru/images/search?p={i}&from=tabbar&text=cat&lr=51&rpt=image", "//img[@class='serp-item__thumb justifier__thumb']","cat")
            time.sleep(10)
    for i in range(6):        
        if len(list_dog) < 1000:
            make_driver(f"https://yandex.ru/images/search?p={i}&from=tabbar&text=dog&lr=51&rpt=image", "//img[@class='serp-item__thumb justifier__thumb']", "dog")
            time.sleep(10)
    save_image()



if __name__ == "__main__":
   
    main()
