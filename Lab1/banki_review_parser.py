import os
import logging
import requests
import argparse
from fake_headers import Headers
from bs4 import BeautifulSoup

BASE_URL = "https://www.banki.ru"
HEADER = Headers(browser="chrome", os="win", headers=True)
TIMEOUTS = (6, 60)


logging.basicConfig(level=logging.INFO)


def create_folder(name:str) -> None:
    """This function creates a new folder if it does not exist"""
    try:
        if not os.path.exists(f"{name}"):
            os.makedirs(name)
    except Exception as exc:
        logging.exception(f"Can not create folder: {exc.message}\n{exc.args}\n")


def get_page(URL: str) -> str:
    """This function decoding url for later use"""
    html_page = requests.get(URL, headers=HEADER.generate(), timeout=TIMEOUTS)
    soup = BeautifulSoup(html_page.text, "lxml")
    return soup


def get_review_links(URL:str) -> list[str]:
    """This function gets all review links from the page"""
    soup = get_page(URL)
    links = soup.findAll('div', "l22dd3882")
    review_links = []
    for link in links:
        review_links.append(f"{BASE_URL}{link.find('a').get('href')}")
    return review_links


def create_txt(count:int, dataset_name: str, rating: int, review: str) -> None:
    """This function creates a new txt file review in corresponding folder"""
    rate_folder = os.path.join(dataset_name, f'{rating}')
    create_folder(rate_folder)
    review_filename = f"{count:04}.txt"
    review_path = os.path.join(rate_folder, review_filename)
    with open(review_path, mode="w", encoding="utf-8") as review_file:
        review_file.write(review)


def review_file(dataset_name: str, link: str, review_count:int) -> None:
    """This function gets review links, checks if the file exists, if not creates a new one"""
    create_folder(dataset_name)
    for rating in range(1, 6):
        page = 1
        count = 0
        while count < review_count:
            url = f"{BASE_URL}/{link}/?page={page}/?type=all&rate={rating}"
            review_links = get_review_links(url)
            for review_link in review_links:
                if not os.path.exists(os.path.join(os.path.join(dataset_name, f'{rating}'), f"{count:04}.txt")):
                    review = get_page(review_link).find('div', class_='lfd76152f').text.strip()
                    if review:
                        try:
                            create_txt(count, dataset_name, rating, review)
                            count +=1
                        except Exception as exc:
                            logging.exception(f"Error downloading review:{exc.args}\n")
                else:
                    count += 1
                    if count % 25 == 0:
                        page += 1
                if count == review_count:
                    break
            logging.info(f"Review {count-25:04}-{count-1:04} has been downloaded")
            page += 1
        logging.info(f"All reviews for {rating} rating has been downloaded")


if __name__=="__main__":
    """This soft is designed to download reviews of banks from https://banki.ru"""
    parser = argparse.ArgumentParser(description='Input path name for reviews, link for parsing, count of reviews')
    parser.add_argument('--path', type=str,default="dataset", help='Input path name for reviews')
    parser.add_argument('--link',type=str,default="services/responses/bank/alfabank" ,help='Input link of the reviews')
    parser.add_argument('--count',type=int,default=500, help='Input count of reviews')
    args = parser.parse_args()
    review_file(args.path, args.link, args.count)