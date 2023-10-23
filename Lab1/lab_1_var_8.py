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

parser = argparse.ArgumentParser(description='Сбор рецензий с КиноПоиск')
parser.add_argument('--max_reviews', type=int, default=1000, help='Максимальное количество рецензий (good и bad) для сбора')
args = parser.parse_args()

try:
    for folder_name in ["good", "bad"]:
        folder_path = os.path.join("dataset", folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
except Exception:
    logging.exception(f"Ошибка при создании папки")

def get_page(page) -> BeautifulSoup:
    base_url = "https://www.kinopoisk.ru/film/435/reviews/ord/rating/status/all/perpage/10/page/"
    url = f"{base_url}{page}/"
    try:
        sleep_time = random.uniform(5, 7)
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

def process_review(review_block, good_reviews, bad_reviews):
    try:
        text_element = review_block.find('div', class_="brand_words")
        if text_element is not None:
            review_text = text_element.get_text()
        else:
            review_text = "Текст рецензии не найден"
            print(review_text)
        element = review_block.find('ul', class_='voter')
        if element:
            li_elements = element.find_all('li')
            if len(li_elements) >= 2:
                second_li = li_elements[1]
                a_element = second_li.find('a')
                if a_element is not None:
                    category = "bad"
                else:
                    category = "good"

                if category == "good":
                    folder_name = "good"
                    good_reviews_count = len(os.listdir(os.path.join("dataset", folder_name)))
                    unique_id = str(good_reviews_count + 1).zfill(4)
                else:
                    folder_name = "bad"
                    bad_reviews_count = len(os.listdir(os.path.join("dataset", folder_name)))
                    unique_id = str(bad_reviews_count + 1).zfill(4)
                file_name = f"{unique_id}_{category}.txt"
                file_path = os.path.join("dataset", folder_name, file_name)
                film_title_element = soup.find('div', class_="breadcrumbs__sub")
                if film_title_element is not None:
                    film_title = film_title_element.get_text()
                else:
                    film_title = "Название фильма не найдено"
                    print(film_title)
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(f"Film: {film_title}\n")
                    file.write(review_text)
    except Exception:
        print(f"Ошибка при обработке рецензии")
    return good_reviews, bad_reviews

good_reviews = 0
bad_reviews = 0
max_reviews_for_one_film = args.max_reviews
page_number = 1
while good_reviews < max_reviews_for_one_film or bad_reviews < max_reviews_for_one_film:
    soup = get_page(page_number)
    if soup:
        review = soup.find('div', class_="clear_all")
        if review is not None:
            review_blocks = review.find_all('div', class_="reviewItem userReview")
            for review_block in review_blocks:
                good_reviews, bad_reviews = process_review(review_block, good_reviews, bad_reviews)
            page_number += 1
        else:
            print("Ошибка при отправке запроса на страницу.")

print(f"Собрано {good_reviews + 1} хороших рецензий и {bad_reviews + 1} плохих рецензий.")

    
