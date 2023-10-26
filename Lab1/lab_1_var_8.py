import logging
import os
import random
from time import sleep

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from typing import List

# Функция для генерации случайного User-Agent
def generate_random_user_agent() -> str:
    """
    Генерирует случайный User-Agent.
    """
    ua = UserAgent()
    return ua.random

logging.basicConfig(level=logging.INFO)

# Функция для создания папок "good" и "bad"
def create_directories():
    """
    Создает директории "good" и "bad" в папке "dataset" (если их нет).
    """
    try:
        for folder_name in ["good", "bad"]:
            folder_path = os.path.join("dataset", folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
    except Exception as e:
        logging.exception(f"Ошибка при создании папки: {e.args}")

create_directories()

# Функция для получения страницы и парсинга её в объект BeautifulSoup
def get_page(page: int, base_url: str = "https://irecommend.ru/content/internet-magazin-ozon-kazan-0?page=") -> BeautifulSoup:
    """
    Получает страницу с веб-сайта и возвращает объект BeautifulSoup для парсинга.
    """
    try:
        url = f"{base_url}{page}"
        sleep_time = random.uniform(1, 3)
        sleep(sleep_time)
        headers = {
            "User-Agent": generate_random_user_agent()
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup
    except requests.exceptions.RequestException as e:
        logging.exception(f"Ошибка при получении страницы: {e.args}")
        return None
    except Exception as e:
        logging.exception(f"Необработанная ошибка: {e.args}")
        return None

# Функция для получения списка рецензий с объекта BeautifulSoup
def get_list_of_reviews(soup: BeautifulSoup) -> List[BeautifulSoup]:
    """
    Получает список рецензий с объекта BeautifulSoup.
    """
    try:
        reviews = soup.find('ul', class_="list-comments").find_all('li')
        return reviews
    except Exception as e:
        logging.exception("Ошибка при получении списка рецензий:", e)

# Функция для извлечения текста рецензии
def review_text(review: BeautifulSoup) -> str:
    """
    Извлекает текст из объекта BeautifulSoup рецензии.
    """
    try:
        text_element = review.find('div', class_="reviewTextSnippet")
        if text_element is not None:
            return text_element.get_text()
        else:
            return "Текст рецензии не найден"
    except Exception as e:
        logging.exception("Ошибка при получении текста рецензии:", e)

# Функция для определения статуса рецензии (хорошая или плохая)
def status_review(review: BeautifulSoup) -> str:
    """
    Определяет статус рецензии (хорошая или плохая) на основе числа звёзд. 
    """
    try:
        stars = review.find_all(class_='on')
        count = len(stars)
        if count > 3:
            return 'good'
        else:
            return 'bad'
    except Exception as e:
        logging.exception("Ошибка при получении статуса рецензии:", e)

# Функция для сохранения рецензии в файл
def save_review_to_file(review_text: str, status_review: str, review_number_good: int, review_number_bad: int):
    """
    Сохраняет рецензию в файл в соответствующей директории (good или bad).
    """
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
        logging.exception(f"Ошибка при сохранении рецензии : {e}")

if __name__ == "__main__":
    number = 0
    review_number_bad = 1
    review_number_good = 1
    for page in range(1, 3):
        reviews = get_list_of_reviews(get_page(page))
        for review in reviews:
            text = review_text(review)
            status = status_review(review)
            save_review_to_file(text, status, review_number_good, review_number_bad)
            if status == 'good':
                review_number_good += 1
                number += 1
            else:
                review_number_bad += 1
                number += 1
        print(number)

