import requests
import os
import re
import logging
import argparse
from time import sleep
from bs4 import BeautifulSoup
import chardet


HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

logging.basicConfig(level=logging.INFO)


def get_page(URL: str) -> str:
    '''Takes in URL of a page, returns a soup for further parsing'''
    sleep(5)
    html_page = requests.get(URL, headers=HEADERS)
    encoding = chardet.detect(html_page.content)['encoding'] #Определяем кодировку страницы
    decoded_html_page = html_page.content.decode(encoding) #Декодируем страницу. Иначе кириллические символы вызовут UnicodeError
    soup = BeautifulSoup(decoded_html_page, features="html.parser")
    return soup


def create_dataset(name: str) -> None:
    '''Takes in a name of dataset directory, creates the directory if it is missing'''
    if not os.path.exists(name):

        try:
            os.mkdir(name)
            for i in range(0, 6):
                os.mkdir(f"{name}/{i}")
        except Exception as exc:
            logging.exception(f"Can't create folder: {exc.message}\n{exc.args}\n")


def parse_soup(soup: str, dataset_name: str, card_nums: dict[str, int]) -> dict[str, int]:
    '''Parses a given soup string and saves review entries into dataset, returns the size of every subdirectory of dataset'''
    cards = soup.find_all("article", class_="review-card lenta__item") #Находим все рецензии

    for card in cards:

        try: #В try паковать его приходится, потому что не везде могут найтись нужные элементы
            rating = card.find("span", class_="lenta-card__mymark").text #Находим оценку книги
        except(AttributeError):
            logging.error("Missing <span> element")
            continue
        rating = re.search(r"\d", rating).group(0)

        try:
            title = card.find("a", class_="lenta-card__book-title").text #Находим название книги
        except(AttributeError):
            logging.error("Missing <a> element")
            continue
        title = str(card.find("a", class_="lenta-card__book-title").text).strip()

        full_text_p = card.find("div", class_="lenta-card__text without-readmore").find("div").find_all("p")
        full_text = "\n"
        for p in full_text_p[1:]: #Здесь 0-й элемент - повтор первого параграфа, поэтому 0-й отбрасываем
            full_text += (p.text + "\n")

        with open(os.path.join(dataset_name, rating, f"{str(card_nums[rating]).zfill(4)}.txt"), "w", encoding="utf-8") as f:
            f.write(title + full_text)
        card_nums[rating] += 1
    
    return card_nums


def circle_pages(url: str, dataset_name: str, target_size: int) -> None:
    '''Takes in a URL, a name of dataset and a target size of the dataset'''
    i = 1
    dataset_size = 0
    card_nums = {str(key):0 for key in range (0,6)}
    while dataset_size < target_size: 
        soup = get_page(url.replace('{page}', str(i)))
        card_nums = parse_soup(soup, dataset_name, card_nums)
        dataset_size = sum(value for value in card_nums.values())
        i += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Save review entries into the dataset')
    parser.add_argument('--url', type=str, default="https://www.livelib.ru/reviews/~{page}#reviews", help='URL of the site to scrap. Replace page number with {page}')
    parser.add_argument('--ds', type=str, default="dataset", help='Name of the dataset directory')
    parser.add_argument('--size', type=int, default=1000, help='Target size of the dataset')
    args = parser.parse_args()

    create_dataset(args.ds)
    circle_pages(args.url, args.ds, args.size)
    