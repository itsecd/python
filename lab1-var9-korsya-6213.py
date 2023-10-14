from bs4 import BeautifulSoup
import os
import requests
from fake_headers import Headers
from time import sleep


def get_html_source(url):
    headers = Headers(browser='chrome', os='win', headers=True).generate()
    source = requests.get(url, headers)
    return source


def get_links_for_all_review(source):
    soup = BeautifulSoup(source.text, 'lxml')
    responses = soup.findAll('div', class_="l22dd3882")
    links=[]
    for resp in responses:
        links.append("https://www.banki.ru" + resp.find('a').get('href'))
    return links

def get_all_review(link):
    source=get_html_source(link)
    soup = BeautifulSoup(source.text, 'lxml')
    all_review = soup.find('p')
    return all_review 

def main():
    url = f"https://www.banki.ru/services/responses/bank/alfabank/?page=1&rate=1"
    source = get_html_source(url)
    sleep(3)
    links=get_links_for_all_review(source)
    #print(get_all_review(links[0]).text)
    #print(links[0])

    '''
    os.mkdir("dataset")
    for num_of_stars in range(1, 6):
        os.mkdir(f"dataset//{num_of_stars}")
        num_of_rev = 0
        for num_of_page in range(1, 28 + 1):
            url = f"https://www.banki.ru/services/responses/bank/alfabank/?page={num_of_page}&rate={num_of_stars}"
            source = get_html_source(url)
            sleep(3)
            responses = soup_search(source)
            for i in range(0, 25):
                num_of_rev += 1
                str_num_of_rev = str(num_of_rev)
                with open(f"dataset//{num_of_stars}//{str_num_of_rev.zfill(4)}.txt", mode='w', encoding='utf-8') as f:
                    f.write(responses[i])
    '''

if __name__ == '__main__':
    main()
