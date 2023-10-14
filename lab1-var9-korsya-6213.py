from bs4 import BeautifulSoup
import requests
from fake_headers import Headers
import time

def get_html_source(url):
    headers=Headers(browser='chrome',os='win',headers=True).generate()
    source=requests.get(url,headers)
    return source

def soup_search(sourse):
    soup=BeautifulSoup(sourse.text,'lxml')
    response_text=soup.find('div',class_="l22dd3882").text
    return response_text

def main():
    url="https://www.banki.ru/services/responses/bank/alfabank/?page=3&rate=2"
    source=get_html_source(url)
    response_text=soup_search(source)
    print(response_text)

if __name__=='__main__':
    main()