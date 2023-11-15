from typing import List, Any
import requests
from bs4 import BeautifulSoup
from time import sleep
import os
import random
import logging
from fake_useragent import UserAgent



def user_interface()->str:
    urls = []
    try:
        print('Введите количество фильмов , на которые хотите получить рецензии: ')
        url_quantity = int(input())
        for i in range(url_quantity):
            print('Введите', i+1, 'ссылку: ')
            url1 = str(input())
            urls.append(url1)
        return(urls)
    except requests.exceptions.RequestException as e:
        logging.exception(f"Введены некоректные данные")
        return None
def get_reviews(soup:BeautifulSoup)->List[BeautifulSoup]:
    try:
            reviews = soup.find('ul', class_="list-comments").find_all('li')
            return reviews
    except Exception as e:
        logging.exception("Ошибка при получении списка рецензий:")


def random_user_agent()->str:
    u = UserAgent()
    return u.random
def get_html_code(page:int , url:str)->BeautifulSoup:
    try:
        tmp_url = f"{url}{page}"
        sleep_time = random.uniform(1, 3)
        sleep(sleep_time)
        headers = {
            "User-Agent": random_user_agent()
        }
        tmp_res = requests.get(tmp_url, headers=headers)
        tmp_res.raise_for_status()
        soup = BeautifulSoup(tmp_res.content, "html.parser")
        return soup
    except requests.exceptions.RequestException as e:
        logging.exception(f"Ошибка при получении html кода")
        return None
if __name__ == "__main__":
        urls= []
        urls = user_interface()
        for i in range (len(urls)):
            print(get_html_code(1,urls[i]))