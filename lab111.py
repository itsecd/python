from typing import List, Any
import requests
from bs4 import BeautifulSoup
from time import sleep
import os
import random
import logging
from fake_useragent import UserAgent



def user_interface() -> list:
    urls: list = []
    try:
        print('Введите количество вещей, на которые хотите получить рецензии: ')
        url_quantity = int(input())
        for i in range(url_quantity):
            print('Введите ссылка на ', i+1, '-юу вещь: ')
            url1 = str(input())
            urls.append(url1)
        return(urls)
    except requests.exceptions.RequestException as e:
        logging.exception(f"Введены некоректные данные")
        return None


def set_pages() -> int:
    print("Со скольки страниц брать рецензии?:")
    tmp_page=int(input())
    return tmp_page


def get_html_code(page: int , url: str) -> BeautifulSoup:
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
    try:
            reviews = soup.find('ul', class_="list-comments").find_all('li')
            return reviews
    except Exception as e:
        logging.exception("Ошибка при получении списка рецензий:")


def review_text(review: BeautifulSoup) -> str:
    try:
        review_txt = review.find('div', class_="reviewTextSnippet")
        if review_txt is not None:
            return review_txt.get_text()
        else:
            return "Текст рецензии не найден"
    except Exception as e:
        logging.exception("Ошибка при получении текста рецензии:", e)


def get_name(soup:BeautifulSoup) -> str:
    try:
        name_txt = soup.find('h1',class_="largeHeader")
        if name_txt is not None:
            return name_txt.get_text()
        else:
            return "Текст названия не найден"
    except Exception as e:
        logging.exception("Ошибка при получении текста названия:", e)


def status_review(review: BeautifulSoup) -> bool:
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


def create_directories():
    try:
        for folder_name in ["good", "bad"]:
            folder_path = os.path.join("dataset", folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
    except Exception as e:
        logging.exception(f"Ошибка при создании папки: {e.args}")


def save_review_to_file(review_text: str, status_review: bool, review_number_good: int, review_number_bad: int, output_dir: str, name: str):
    if status_review :
        folder_name = "good"
        file_name = f"{review_number_good:04d}.txt"
    else:
        folder_name = "bad"
        file_name = f"{review_number_bad:04d}.txt"

    file_path = os.path.join(output_dir, folder_name, file_name)

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(name)
            file.write(review_text)
    except Exception as e:
        logging.exception(f"Ошибка при сохранении рецензии : {e}")


if __name__ == "__main__":
        urls: list = user_interface()
        pages: int = set_pages()
        create_directories()
        output_dir: str = "dataset"
        number = 0
        review_n_b = 1
        review_n_g = 1
        for url in urls:
            name: str = get_name(get_html_code(1, url))
            for page in range(pages):
                rev = get_reviews(get_html_code(page,url))
                for review in rev:
                    txt = review_text(review)
                    status = status_review(review)
                    save_review_to_file(txt, status, review_n_g, review_n_b, output_dir, name)
                    if status == 'good':
                        review_n_g += 1
                        number += 1
                    else:
                        review_n_b += 1
                        number += 1
        print("Было успешно сохранено", number ,"рецензий")

