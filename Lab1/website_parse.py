from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from time import sleep
import os
page = 2

count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}


def getting_full_review():
    sleep(5)
    buttons = driver.find_elements(By.CSS_SELECTOR, "a.read-more__link")


while sum(count) <= 6000:
    url = f"https://www.livelib.ru/reviews/~{page}#reviews"
    driver = webdriver.Chrome()
    driver.get(url)
    page += 1
