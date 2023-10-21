import requests
from bs4 import BeautifulSoup
import os
from time import sleep
import logging
import random
from fake_useragent import UserAgent


def generate_random_user_agent():
    ua = UserAgent()
    return ua.random

logging.basicConfig(level=logging.INFO)

try:
    if not os.path.exists("dataset/good"):
        os.makedirs("dataset/good")
    if not os.path.exists("dataset/bad"):
        os.makedirs("dataset/bad")
except Exception:
    logging.exception(f"Ошибка при создании папки")

def get_page(page):
    url = "https://www.kinopoisk.ru/film/435/reviews/ord/rating/status/all/perpage/10/page/" + str(page) + "/"
    try:
        sleep_time = random.uniform(1, 3)
        sleep(sleep_time)
        headers = {
            "User-Agent": generate_random_user_agent()  # Используем случайный User-Agent
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
        # Получаем текст рецензии
        text_element = review_block.find('div', class_="brand_words")
        if text_element is not None:
            review_text = text_element.get_text()
        else:
            review_text = "Текст рецензии не найден"
            print(review_text)
        # Определяем, является ли рецензия "good" или "bad"
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

                # Создаем уникальный идентификатор в виде числа от 0001 до 0999
                if category == "good":
                    good_reviews = len(os.listdir("dataset/good"))
                    unique_id = str(good_reviews + 1).zfill(4)
                else:
                    bad_reviews = len(os.listdir("dataset/bad"))
                    unique_id = str(bad_reviews + 1).zfill(4)

                # Создаем путь к файлу, включая папку с категорией
                file_name = f"{unique_id}_{category}.txt"
                file_path = os.path.join("dataset", category, file_name)
                film_title_ = soup.find('div',class_="breadcrumbs__sub")
                if film_title_ is not None:
                    film_title = film_title_.get_text()
                else:
                    film_title = "Название фильма не найдено"
                    print(film_title)

                # Записываем информацию в файл
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(f"Film: {film_title}\n")
                    file.write(review_text)
    except Exception:
        print(f"Ошибка при обработке рецензии")

    return good_reviews, bad_reviews

good_reviews = 0
bad_reviews = 0
max_reviews_for_one_film = 10
page_number = 2

while good_reviews < max_reviews_for_one_film or bad_reviews < max_reviews_for_one_film:
    soup = get_page(page_number)
    if soup:
        review = soup.find('div', class_="clear_all")
        if review is not None:
            review_blocks=review.find_all('div',class_="reviewItem userReview")
            for review_block in review_blocks:
                good_reviews, bad_reviews = process_review(review_block, good_reviews, bad_reviews)
            page_number += 1
        else:
            print("Ошибка при отправке запроса на страницу.")


print(f"Собрано {good_reviews+1} хороших рецензий и {bad_reviews+1} плохих рецензий.")    

    
