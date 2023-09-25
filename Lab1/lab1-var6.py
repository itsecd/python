import os
import requests
from bs4 import BeautifulSoup


def download_images(my_list, folder_name):
    os.mkdir(f'dataset/{folder_name}')
    num = 0
    for i in my_list:
        response = requests.get(i)
        with open(f"dataset/{folder_name}/{format(num).zfill(4)}.jpg", 'wb') as f:
            f.write(response.content)
        num += 1


def correct_images(my_list, folder_name):
    print('total: ', len(my_list))
    sorted_list = list(set(my_list))
    print('total: ', len(sorted_list))
    download_images(sorted_list, folder_name)


def create_list(split, folder_name):
    list = []
    count = 0
    for pages in range(1, 10):
        url = f"https://www.yandex.ru/images/search?lr=51&p={pages}&rpt=image&{split}"
        response = requests.get(url).text
        soup = BeautifulSoup(response, 'lxml')
        images = soup.find_all('img')
        for image in images[1:]:
            count += 1
            list.append("https:"+image['src'])
    correct_images(list, folder_name)

os.mkdir('dataset')
url_leopard = 'https://yandex.ru/images/search?text=leopard'
split_leopard = (url_leopard.split('?'))[1]
folder_name_leopard = (url_leopard.split('='))[1]
create_list(split_leopard, folder_name_leopard)
url_leopard = 'https://yandex.ru/images/search?text=tiger'
split_leopard = (url_leopard.split('?'))[1]
folder_name_tiger = (url_leopard.split('='))[1]
create_list(split_leopard, folder_name_tiger)
