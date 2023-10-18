import bs4
import requests
import csv
from datetime import datetime

url = 'https://www.gismeteo.ru/diary/4618/'


def get_url_with_current_date() -> str:
    '''
    Эта функция возвращает дату типа "2023/10" для того, чтобы запрашивать
    актуальный месяц погоды
    '''
    date = datetime.strptime(str(datetime.now().date()), "%Y-%m-%d")
    return '/'.join([str(date.year),str(date.month)])


def get_data(url: str) -> str:
    '''
    Эта функция делает GET запрос и возвращает html с сайта погоды
    '''
    headers = requests.utils.default_headers()

    headers.update(
        {
            'User-Agent': 'My User Agent 1.0',
        }
    )

    return requests.get(url=url, headers=headers).text


def get_rows_of_weather_table(data: str) -> bs4.element.ResultSet:
    '''
    Эта функция принимает на вход html и возвращает список найденых элементов tr
    с атрибутами align=center
    '''
    soup = bs4.BeautifulSoup(data, 'html.parser')
    return soup.find_all("tr", {"align": "center"})


def write_weather_csv(url: str):
    '''
    Эта функция запрашивает и инициалезирует остальные функции и формирует csv файл
    '''

    url = url+get_url_with_current_date()+'/'
    data = get_data(url=url)
    rows = get_rows_of_weather_table(data=data)

    header = [
        'Дата',
        'Температура',
        'Давление',
        'Ветер'
        ]

    with open('weather.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in rows:
            writer.writerow([j.get_text() for j in i.find_all('td') if j.get_text() != ""][:4])

if __name__ == '__main__':
    write_weather_csv(url=url)