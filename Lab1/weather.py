# -*- coding: utf-8 -*-

import bs4
import requests
import csv
from datetime import datetime
from dateutil.relativedelta import *
from fake_headers import Headers
import logging
import argparse
import os

URL = 'https://www.gismeteo.ru/diary/'

HEADER = [
        'Дата',
        'Температура',
        'Давление',
        'Ветер'
        ]


def get_list_of_date(start_date: datetime, end_date: datetime) -> list:
    '''
    Эта функция возвращает список с датами типа ['2008/01', '2008/02', '2008/03'], 
    чтобы запрашивать актуальный месяц погоды
    '''
    list_of_date = list()
    while start_date < end_date:
        list_of_date.append(start_date.strftime('%Y/%m'))
        start_date = start_date + relativedelta(months=+1)

    return list_of_date


def get_data(url: str) -> str:
    '''
    Эта функция делает GET запрос и возвращает html с сайта погоды
    '''
    try:
        header = Headers(
            browser="chrome",
            os="win",
            headers=True
            )
        return requests.get(url=url, headers=header.generate()).text
    except Exception as e:
        logging.error(e)


def get_rows_of_weather_table(data: str, date: str) -> bs4.element.ResultSet:
    '''
    Эта функция принимает на вход html и возвращает список найденых элементов tr
    с атрибутами align=center
    '''
    try:
        logging.info(f"Выкачиваем данные за {date}")
        soup = bs4.BeautifulSoup(data, 'html.parser')
        rows = soup.find_all("tr", {"align": "center"})
        ans = list()
        for i in rows:
            row = [j.get_text() for j in i.find_all('td') if j.get_text() != ""][:4]
            row[0] = f"{date}/{row[0]}"
            ans.append(row)
        return ans
    except Exception as e:
        logging.error(e)


def write_weather_csv(code_sity: str, path: str, start_date: datetime, end_date: datetime):
    '''
    Эта функция запрашивает и инициализирует остальные функции и формирует csv файл
    '''
    dates = get_list_of_date(start_date, end_date)
    
    months = list()
    for i in dates:
        data = get_data(url=f"{URL}{code_sity}/{i}/")
        rows = get_rows_of_weather_table(data=data, date=i)
        months.append(rows)

    try:
        if not os.path.exists(path):
            os.mkdir(path)
        fullname = os.path.join(path, 'weather.csv')

        with open(file=fullname, mode='w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for rows in months:
                for i in rows:
                    writer.writerow(i)
    except Exception as e:
        logging.error(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My example explanation')
    parser.add_argument(
        '-code_sity',
        type=str,
        default='4618',
        help='provide an string (default: 4618)'
    )
    parser.add_argument(
        '-path',
        type=str,
        default='./',
        help='provide an string (default: ./)'
    )
    parser.add_argument(
        '-start_date',
        type=datetime,
        default=datetime(2008, 1, 1),
        help='provide an datetime (default: datetime(2008, 1, 1))'
    )
    parser.add_argument(
        '-end_date',
        type=datetime,
        default=datetime.now(),
        help='provide an datetime (default: datetime.now())'
    )
    namespace = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    write_weather_csv(code_sity=namespace.code_sity, 
                      path=namespace.path, 
                      start_date=namespace.start_date, 
                      end_date=namespace.end_date)
