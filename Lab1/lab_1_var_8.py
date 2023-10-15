import requests
from bs4 import BeautifulSoup
import os
import re

# Создаем папки для хранения рецензий
if not os.path.exists("dataset/good"):
    os.makedirs("dataset/good")
if not os.path.exists("dataset/bad"):
    os.makedirs("dataset/bad")

good_reviews = 0
bad_reviews = 0
max_reviews_for_one_film = 300
page_number = 1
film_reviews = "https://www.rottentomatoes.com/m/the_equalizer_3/reviews?type=top_critics"
while good_reviews < max_reviews_for_one_film or bad_reviews < max_reviews_for_one_film:
        response = requests.get(film_reviews, params={"page": page_number})
        soup = BeautifulSoup(response.text, "html.parser") 
        match = re.search(r"/m/(.*?)/reviews", film_reviews)
        movie_title = match.group(1)
        print(movie_title)    
        # Находим блоки с рецензиями 
        review_blocks = soup.find_all('div',class_="review-row")
        for review_block in review_blocks:
            # Получаем текст рецензии
            review_text = review_block.find('p', class_="review-text").get_text()
            # Определяем, является ли рецензия "good" или "bad"
            element = review_block.find('score-icon-critic-deprecated')
            category=element["state"]
            if category == "rotten":
                category = "Bed"
                print(review_text,"///////////",category)
            else:
                category = "Good"
                print(review_text,"///////////",category)
            # Создаем путь для сохранения файла
            review_number = good_reviews + bad_reviews
            file_path = f"dataset/{category}/{str(review_number).zfill(4)}.txt"
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(f"Movie: {movie_title}\n")
                file.write(review_text)
                        
            if category == "good":
                good_reviews += 1
            else:
                bad_reviews += 1
        page_number += 1    

    
