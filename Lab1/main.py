import requests
from bs4 import BeautifulSoup
import os


# Функция для получения страницы с рецензиями по ссылке
def get_reviews_page(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Не удалось получить страницу: {url}")
        return None

# Функция для парсинга рецензий на странице
def parse_reviews(page_content):
    soup = BeautifulSoup(page_content, 'lxml')
    reviews = soup.find_all('div', class_='discovery-tiles__wrap')
    return reviews

# Создаем папки для хранения рецензий
os.makedirs('dataset/good', exist_ok=True)
os.makedirs('dataset/bad', exist_ok=True)

url = "https://www.rottentomatoes.com/m/the_equalizer_3/reviews?type=user"
r = requests.get(url)
soup = BeautifulSoup(r.text,"lxml")
film_links = soup.find_all('div')
# Счетчики для файлов
good_count = 0
bad_count = 0

# Цикл по ссылкам на фильмы
for film_link in film_links:
    # Получаем страницу с рецензиями
    page_content = get_reviews_page(film_link)
    if page_content:
        # Парсим рецензии
        reviews = parse_reviews(page_content)
        # Определяем категорию рецензии (по ссылке)
        if "status/bad" in film_link:
            category = "bad"
        else:
            category = "good"

        # Сохраняем рецензии в файлы
        for review in reviews:
            # Получаем название фильма
            film_title = review.find('span', class_='p--small').text.strip()
            # Определяем номер файла и формируем имя файла
            if category == "bad":
                filename = f'dataset/bad/{str(bad_count).zfill(4)}.txt'
                bad_count += 1
            else:
                filename = f'dataset/good/{str(good_count).zfill(4)}.txt'
                good_count += 1

            # Сохраняем рецензию в файл
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(film_title + '\n')
                file.write(review.text)
print("Рецензии сохранены в папках dataset/good и dataset/bad.")