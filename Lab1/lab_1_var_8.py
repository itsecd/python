import requests
from bs4 import BeautifulSoup
import os
from time import sleep
import logging
import random
from fake_useragent import UserAgent
import argparse

def generate_random_user_agent():
    ua = UserAgent()
    return ua.random

logging.basicConfig(level=logging.INFO)


def create_directories():
    try:
        for folder_name in ["good", "bad"]:
            folder_path = os.path.join("dataset", folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
    except Exception:
        logging.exception(f"Ошибка при создании папки")
create_directories()
 
def get_page(page) -> BeautifulSoup:
    url ="https://irecommend.ru/content/internet-magazin-ozon-kazan-0?page="+ str(page)
    try:
        sleep_time = random.uniform(1, 3)
        sleep(sleep_time)
        headers = {
            "User-Agent": generate_random_user_agent()
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup
    except requests.exceptions.RequestException:
        logging.error(f"Ошибка при получении страницы")
        return None
    except Exception:
        logging.error(f"Необработанная ошибка")
        return None
def get_list_of_reviews(soup):
    try:
        reviews = soup.find('ul', class_="list-comments").find_all('li')
        return reviews
    except Exception as e:
        logging.error("Ошибка при получении списка рецензий:", e)

def review_text(review):
    try:
        text_element = review.find('div', class_="reviewTextSnippet")
        if text_element is not None:
            return text_element.get_text()
        else:
            return "Текст рецензии не найден"
    except Exception as e:
        logging.error("Ошибка при получении текста рецензии:", e)

def status_review(review):
    try:
        stars = review.find_all(class_='on')
        count = len(stars)
        if count>3:
            return 'good'
        else:
            return 'bad'
    except Exception as e:
        logging.error("Ошибка при получении статуса рецензии:", e)

def save_review_to_file(review_text, status_review,review_number_good,review_number_bad):
    if status_review == "good":
        folder_name = "good"
        file_name = f"{review_number_good:04d}.txt"
        file_path = os.path.join("dataset", folder_name, file_name) 
    else:
        folder_name = "bad"
        file_name = f"{review_number_bad:04d}.txt"
        file_path = os.path.join("dataset", folder_name, file_name)

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(review_text)
    except Exception as e:
        logging.error(f"Ошибка при сохранении рецензии : {e}")

number = 1
review_number_bad = 1
review_number_good = 1
for page in range(1,61):     
    reviews = get_list_of_reviews(get_page(page))
    for review in reviews:
        text = review_text(review)
        status = status_review(review)
        save_review_to_file(text, status, review_number_good, review_number_bad)
        if status == 'good':
            review_number_good += 1
            number+=1
        else:
            review_number_bad += 1
            number+=1
    print(number)