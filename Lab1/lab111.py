import logging
import os
import random
from time import sleep

import argparse
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from typing import List

FOLDER_NAME_G = "good"
FOLDER_NAME_B = "bad"

def parse_arguments() -> argparse:
    """
            Получаем ссылку путь директории и количество страниц

    """
    parser = argparse.ArgumentParser(description="Скрипт для парсинга рецензий с сайта и сортировка их на хорошии и плохие")
    parser.add_argument("--out_dir", type=str, default="dataset", help="Путь к директории для сохранения датасета")
    parser.add_argument("--urls", type=str, default="https://irecommend.ru/content/internet-magazin-ozon-kazan-0?page=", help="Базовый URL для сбора данных")
    parser.add_argument("--pages", type=int, default=3, help="Количество страниц для обхода")
    return parser.parse_args()


def get_html_code(page: int , url: str) -> BeautifulSoup:
    """
               Получаем html код страницы.

    """

    try:
        tmp_url = url+str(page)
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


def get_reviews(soup: BeautifulSoup) -> List[BeautifulSoup]:
    """
            Создаем лист из рецензий на одной страници

    """
    try:
            reviews = soup.find('ul', class_="list-comments").find_all('li')
            return reviews
    except Exception as e:
        logging.exception("Ошибка при получении списка рецензий:")


def review_text(review: BeautifulSoup) -> str:
    """
            Получасем текс рецензий

    """
    try:
        review_txt = review.find('div', class_="reviewTextSnippet")
        if review_txt is not None:
            return review_txt.get_text()
        else:
            return "Текст рецензии не найден"
    except Exception as e:
        logging.exception("Ошибка при получении текста рецензии:", e)


def get_name(soup:BeautifulSoup) -> str:
    """
            Получасем название обьекта на которые парсим рецензии

    """
    try:
        name_txt = soup.find('h1',class_="largeHeader")
        if name_txt is not None:
            return name_txt.get_text()
        else:
            return "Текст названия не найден"
    except Exception as e:
        logging.exception("Ошибка при получении текста названия:", e)


def status_review(review: BeautifulSoup) -> bool:
    """
        Получаем статус рецензии

    """
    try:
        stars = review.find_all(class_='on')
        if len(stars) > 3:
            return True
        else:
            return False
    except Exception as e:
        logging.exception("Ошибка при получении статуса рецензии:", e)


def random_user_agent() -> str:
    u = UserAgent()
    return u.random


def create_directories(page: int, urls: str, out_dir: str) -> None:
    """
        Создаем директорию для сохранения рецензий если он еще не создана

    """
    try:
        folder_path_g = os.path.join(args.out_dir, FOLDER_NAME_G)
        folder_path_b = os.path.join(args.out_dir, FOLDER_NAME_B)
        if not os.path.exists(folder_path_g):
            os.makedirs(folder_path_g)
        if not os.path.exists(folder_path_b):
            os.makedirs(folder_path_b)
    except Exception as e:
        logging.exception(f"Ошибка при создании папки: {e.args}")


def save_review_to_file(review_text: str, status_review: bool, review_n_g: int, review_n_b: int) -> None:
    """
    Сохраняем полученный текс ревью в файл нужной директории в зависимости хорошая или плохая рецензия

    """
    if status_review:
        file_name = f"{review_n_g:04d}.txt"
        file_path = os.path.join(args.out_dir, FOLDER_NAME_G, file_name)
    else:
        file_name = f"{review_n_b:04d}.txt"
        file_path = os.path.join(args.out_dir, FOLDER_NAME_B, file_name)

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(review_text)
    except Exception as e:
        logging.exception(f"Ошибка при сохранении рецензии : {e}")


def parsing_review(page: int, urls: str, out_dir: str) -> None:
    """
    Итоговая функция которая используя все предидущие функции  полностью выполняет задачу

    """
    try:
        number = 0
        review_n_b = 1
        review_n_g = 1
        for page in range(1, args.pages + 1):
            rev = get_reviews(get_html_code(page, args.urls))
            for review in rev:
                txt = review_text(review)
                status = status_review(review)
                save_review_to_file(txt, status, review_n_g, review_n_b)
                if status:
                    review_n_g += 1
                    number += 1
                else:
                    review_n_b += 1
                    number += 1
    except Exception as e:
        logging.exception(f"Ошибка при сохранении рецензии : {e}")


if __name__ == "__main__":
    args = parse_arguments()
    create_directories(*(vars(args).values()))
    parsing_review(*(vars(args).values()))

