from bs4 import BeautifulSoup
import requests
from fake_headers import Headers
from time import sleep

def get_html_source(url):
    headers=Headers(browser='chrome',os='win',headers=True).generate()
    source=requests.get(url,headers)
    return source

def soup_search(sourse):
    soup=BeautifulSoup(sourse.text,'lxml')
    responses=soup.findAll('div',class_="l22dd3882")
    data=[]
    for resp in responses:
        data.append(resp.text.replace('\t','').replace('\n','').replace('\r',''))
    return data

def main():
    for num_of_stars in range(1,6):
        for num_of_page in range(1,2):
            url=f"https://www.banki.ru/services/responses/bank/alfabank/?page={num_of_page}&rate={num_of_stars}"
            source=get_html_source(url)
            sleep(3)
            responses=soup_search(source)
            #print(responses)

if __name__=='__main__':
    main()