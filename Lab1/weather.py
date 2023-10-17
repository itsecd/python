import requests
from bs4 import BeautifulSoup
import json
import csv

url = 'https://www.gismeteo.ru/diary/4618/2023/10/'


def get_data(url):
    headers = requests.utils.default_headers()

    headers.update(
        {
            'User-Agent': 'My User Agent 1.0',
        }
    )

    return requests.get(url, headers=headers)


def get_rows_of_weather_table(data):
    soup = BeautifulSoup(data.text, 'html.parser')
    return soup.find_all("tr", {"align": "center"})


def write_weather_csv():
    data = get_data(url)
    rows = get_rows_of_weather_table(data)

    header = ['Дата', 'Температура', 'Давление', 'Ветер']
    with open('weather.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in rows:
            writer.writerow([j for j in i.find_all('td') if j.get_text() != ""][:4])


if __name__ == '__main__':
    write_weather_csv()
