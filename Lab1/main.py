from bs4 import BeautifulSoup
import requests
import os
import re
import shutil
import codecs
import chardet


#Блок работы с запросом и кодом страницы

URL = "https://www.livelib.ru/reviews/~1#reviews"
html_page = requests.get(URL, headers={"User-Agent":"Mozilla/5.0"})
encoding = chardet.detect(html_page.content)['encoding'] #Определяем кодировку страницы
decoded_html_page = html_page.content.decode(encoding) #Декодируем страницу. Иначе кириллические символы вызовут UnicodeError
soup = BeautifulSoup(decoded_html_page, features="html.parser")
print(soup)

card_nums = {
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0,
    '5': 0,
}

#Блок работы с директориями. 

shutil.rmtree("dataset") #Удаляем старый датасет
os.mkdir("dataset")
for i in range(1, 6):
    os.mkdir(f"dataset/{i}")

#Блок парсинга 

cards = soup.find_all("article", class_="review-card lenta__item") #Находим все рецензии


for card in cards:

    try: #В try паковать его приходится, потому что не везде могут найтись нужные элементы
        rating = card.find("span", class_="lenta-card__mymark").text #Находим оценку книги
    except(AttributeError):
        continue
    rating = re.search(r"\d", rating).group(0)

    try:
        title = card.find("a", class_="lenta-card__book-title").text #Находим название книги
    except(AttributeError):
        continue
    title = str(card.find("a", class_="lenta-card__book-title").text).strip()

    full_text_p = card.find("div", class_="lenta-card__text without-readmore").find("div").find_all("p")
    full_text = "\n"
    for p in full_text_p:
        full_text += (p.text + "\n")

    with open("dataset/" + rating + "/" + str(card_nums[rating]).zfill(4), "w") as f:
        f.write(title + full_text)
    card_nums[rating] += 1
    print(rating)


