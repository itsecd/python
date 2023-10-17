from bs4 import BeautifulSoup
import requests
import os
import re
import codecs
import chardet
import logging


#Блок работы с запросом и кодом страницы
def get_page(URL: str) -> str:
    html_page = requests.get(URL, headers={"User-Agent":"Mozilla/5.0"}, timeout=100)
    encoding = chardet.detect(html_page.content)['encoding'] #Определяем кодировку страницы
    decoded_html_page = html_page.content.decode(encoding) #Декодируем страницу. Иначе кириллические символы вызовут UnicodeError
    soup = BeautifulSoup(decoded_html_page, features="html.parser")
    return soup


#Блок работы с директориями. 
def create_dataset(name: str) -> None:
    if not os.path.exists(f"{name}"):
        os.mkdir(name)
        for i in range(0, 6):
            os.mkdir(f"{name}/{i}")


#Блок парсинга 
def parse_soup(soup: str, dataset_name: str) -> None:
    cards = soup.find_all("article", class_="review-card lenta__item") #Находим все рецензии
    card_nums = {
    '0': 0,
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0,
    '5': 0,
}

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

        with open(f"{dataset_name}/" + rating + "/" + str(card_nums[rating]).zfill(4), "w", encoding="utf-8") as f:
            f.write(title + full_text)
            print("Dataset updated")
        card_nums[rating] += 1
        

URL = "https://www.livelib.ru/reviews/~1#reviews"

create_dataset("dataset")

for i in range(1, 100):
    soup = get_page(f"https://www.livelib.ru/reviews/~{i}#reviews")
    print(f"https://www.livelib.ru/reviews/~{i}#reviews")

    parse_soup(soup, "dataset")











