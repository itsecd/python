from bs4 import BeautifulSoup
from fake_headers import Headers
import os
import requests
from time import sleep


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


def get_links_for_all_review(source) -> list[str]:
    """
    the function finds the end part of the link in html and connects it to
    "https://www.banki.ru"
    Parameters
    ----------
    source : str
    Returns
    -------
    list[str]
    full text of the array of links
    """
    soup = BeautifulSoup(source, 'lxml')
    responses = soup.findAll('div', class_="l22dd3882")
    links = []
    for resp in responses:
        links.append("https://www.banki.ru" + resp.find('a').get('href'))
    return links


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
    func makes a new directory using the passed path
    Parameters
    ----------
    path : str
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as err: 
        logging.error(f"{err}", exc_info=True)

def main():
    mkdir("dataset")
    for num_of_stars in range(1, 6):
        mkdir(f"dataset//{num_of_stars}")
        num_of_rev = 0
        for num_of_page in range(1, 28 + 1):
            url = f"https://www.banki.ru/services/responses/bank/alfabank/?page={num_of_page}&rate={num_of_stars}"
            source = get_html_source(url)
            sleep(3)
            links = get_links_for_all_review(source)
            for link in links:
                review = get_all_review(link)
                if review.replace(
                    "\t",
                        "") != "\nПомогите другим пользователям выбрать лучший банк\n":
                    num_of_rev += 1
                    str_num_of_rev = str(num_of_rev)
                    with open(f"dataset//{num_of_stars}//{str_num_of_rev.zfill(4)}.txt", mode='w', encoding='utf-8') as f:
                        f.write(review)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=os.path.join("py_log.log"), filemode="w")
    main()
