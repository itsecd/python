import logging
import os
import random
from time import sleep

import argparse
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from typing import List


def parse_arguments():
    parser = argparse.ArgumentParser(description="Скрипт для сбора данных с веб-сайта и сохранения их в датасет")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Путь к директории для сохранения датасета")
    parser.add_argument("--base_url", type=str, default="https://irecommend.ru/content/internet-magazin-ozon-kazan-0?page=", help="Базовый URL для сбора данных")
    parser.add_argument("--pages", type=int, default=3, help="Количество страниц для обхода")
    return parser.parse_args()


def generate_random_user_agent() -> str:
    """
    Генерирует случайный User-Agent.
    """
    ua = UserAgent()
    return ua.random

logging.basicConfig(level=logging.INFO)


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


def get_page(page: int, base_url: str = "https://irecommend.ru/content/ofitsialnyi-internet-magazin-parfyumerii-i-kosmetiki-letual?page=") -> BeautifulSoup:
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


def get_list_of_reviews(soup: BeautifulSoup) -> List[BeautifulSoup]:
    """
    Получает список рецензий с объекта BeautifulSoup.
    """
    try:
        reviews = soup.find('ul', class_="list-comments").find_all('li')
        return reviews
    except Exception as e:
        logging.exception("Ошибка при получении списка рецензий:", e)


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


def save_review_to_file(review_text: str, status_review: str, review_number_good: int, review_number_bad: int, output_dir: str):
    """
    Сохраняет рецензию в файл в соответствующей директории (good или bad).
    """
    if status_review == "good":
        folder_name = "good"
        file_name = f"{review_number_good:04d}.txt"
    else:
        folder_name = "bad"
        file_name = f"{review_number_bad:04d}.txt"
    
    file_path = os.path.join(output_dir, folder_name, file_name)

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(review_text)
    except Exception as e:
        logging.exception(f"Ошибка при сохранении рецензии : {e}")

if __name__ == "__main__":
    args = parse_arguments()
    output_dir = args.output_dir
    base_url = args.base_url
    pages = args.pages

    number = 0
    review_number_bad = 0
    review_number_good = 0
    for page in range(1, pages + 1):
        reviews = get_list_of_reviews(get_page(page, base_url))
        for review in reviews:
            text = review_text(review)
            status = status_review(review)
            save_review_to_file(text, status, review_number_good, review_number_bad, output_dir)
            if status == 'good':
                review_number_good += 1
                number += 1
            else:
                review_number_bad += 1
                number += 1
        print(number)

