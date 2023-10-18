import os
from bs4 import BeautifulSoup
import requests
import chardet
import logging
import argparse

BASE_URL = "https://otzovik.com/reviews/" #Пока заглушка
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}


logging.basicConfig(level=logging.INFO)


def create_folder(name:str) -> None: #Создает папки в корневой папке лабы, норм? Может переписать на две функции?
    try:
        if not os.path.exists(f"{name}"):
            os.mkdir(name)
    except Exception as exc:
        logging.exception(f"Can not create folder: {exc.message}\n{exc.args}\n")


def get_page(URL: str) -> str: #Работает нормально, стоит переписать, возможно
    html_page = requests.get(URL, headers=HEADERS, timeout=5)
    encode = chardet.detect(html_page.content)['encoding']
    decoded_html_page = html_page.content.decode(encode)
    soup = BeautifulSoup(decoded_html_page, features="html.parser")
    return soup


def w_review_to_txt_file(soup: str, dataset_name: str, n_of_reviews:int)-> dict[str, int]:
    create_folder(dataset_name)
    for rating in range(1, 6):
        page = 1
        count = 0
        while count < n_of_reviews:
            url = f"{BASE_URL}/{page}/?ratio={rating}"
            soup = get_page(url)
            reviews = soup.find_all('div', itemprop ="review")
            for review in reviews:
                review_url = review["review-teaser"]
                if review_url:
                    try:
                        review_data = requests.get(review_url, headers=HEADERS, timeout=10).content
                        rate_folder = os.path.join(dataset_name, page).replace("\\","/")
                        create_folder(rate_folder)
                        review_filename = f"{count:04}.txt"
                        review_path = os.path.join(rate_folder, review_filename)
                        with open(review_path, "w") as review_file:
                            review_file.write(review_data)
                        count +=1
                        if count >= n_of_reviews:
                            break
                    except Exception as exc:
                        logging.exception(f"Error downloading review:{exc.args}\n")
            page += 1
        logging.info(f"All reviews for {rating} rating has been downloaded")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Input directory path, link for parsing, count of ___")
    parser.add_argument("-p", "--path", help="Input directory path", type=str)
    parser.add_argument("-l", "--link", help="Input link", type=str)
    parser.add_argument("-c", "--count", help="input count of ___")
    args = parser.parse_args()
    w_review_to_txt_file()