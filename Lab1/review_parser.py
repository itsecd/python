import argparse
from bs4 import BeautifulSoup
from fake_headers import Headers
import os
import logging
import requests
from time import sleep
import random


def get_html_source(url: str) -> str:
    """
    the function gets the source code of the page
    Parameters
    ----------
    url : str
    Returns
    -------
    str
    the source code of the page
    """
    headers = Headers(browser='chrome', os='win', headers=True).generate()
    source = requests.get(url, headers).text
    return source


def get_links_for_all_review(
        source: str,
        link="https://www.banki.ru",
        find_class="l22dd3882") -> list[str]:
    """
    the function finds the end part of the link in html and connects it to
    "https://www.banki.ru"
    Parameters
    ----------
    source : str
    link : str
    find_class : str
    Returns
    -------
    list[str]
    full text of the array of links
    """
    soup = BeautifulSoup(source, 'lxml')
    responses = soup.findAll('div', find_class)
    return [f"{link}{resp.find('a').get('href')}" for resp in responses]


def get_all_review(link: str) -> str:
    """
    the function returns the full text of the review
    Parameters
    ----------
    link : str
    Returns
    -------
    str
    full text of the review
    """
    source = get_html_source(link)
    soup = BeautifulSoup(source, 'lxml')
    all_review = soup.find('p').text
    return all_review


def mkdir(path: str) -> None:
    """
    func handles folder creation using exceptions
    Parameters.
    ----------
    path : str
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as err:
        logging.error(f"{err}", exc_info=True)


def dirs_for_reviews(path: str) -> None:
    """
    func сreates the required folder and subfolders for reviews
    Parameters.
    ----------
    path : str
    """
    mkdir(path)
    for num_of_stars in range(1, 6):
        mkdir(os.path.join(f"{path}", f"{num_of_stars}"))


def w_reviews_to_files(
        path_n_dir="dataset",
        url="https://www.banki.ru/services/responses/bank/alfabank/?page=",
        cnt_pages=28)->None:
    """
    func writes the full text of the review to a text file in the desired folder
    Parameters.
    ----------
    path_n_dir : str
    url : str
    cnt_pages : int
    """
    dirs_for_reviews(path_n_dir)
    for num_of_stars in range(1, 6):
        num_of_rev = 0
        for num_of_page in range(1, cnt_pages + 1):
            url = f"{url}{num_of_page}&rate={num_of_stars}"
            source = get_html_source(url)

            sleep(1 if random.random()>0.5 else 2)
            links = get_links_for_all_review(source)
            for link in links:
                review = get_all_review(link)
                if review.replace(
                    "\t",
                        "") != "\nПомогите другим пользователям выбрать лучший банк\n":
                    num_of_rev += 1
                    str_num_of_rev = str(num_of_rev)
                    with open(os.path.join(f"{path_n_dir}", f"{num_of_stars}", f"{str_num_of_rev.zfill(4)}.txt"), mode='w', encoding='utf-8') as f:
                        f.write(review)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=os.path.join("py_log.log"), filemode="w")
    parser = argparse.ArgumentParser(description="Input directory path, link for parse, count of pages")
    parser.add_argument("-p", "--path", help="Input directory path", type=str)
    parser.add_argument("-l", "--link", help="Input link", type=str)
    parser.add_argument("-c", "--count", help="Input count of pages", type=int)
    args = parser.parse_args()
    #w_reviews_to_files(args.path,args.link,args.count)
    w_reviews_to_files()
