# -*- coding: utf-8 -*-

import logging
import argparse
import csv
from datetime import datetime, timedelta
import os

logging.basicConfig(level=logging.INFO)


def read_csv(file_path: str) -> tuple:
    '''
    This function reads a CSV file
    :param file_path:
    :return tuple(header: list(str), data: list(str)):
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)
            data = list(reader)
        return header, data
    except Exception as e:
        logging.exception(f"Can't read data: {e}\n{e.args}\n")


def write_csv(file_path: str, header: list, data: list) -> None:
    '''
    This function writes a CSV file
    :param file_path:
    :param header:
    :param data:
    :return:
    '''
    try:
        with open(file_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data)
    except Exception as e:
        logging.exception(f"Can't write data: {e}\n{e.args}\n")


def split_csv_by_columns(input_file: str, output_file_x: str, output_file_y: str) -> None:
    '''
    This function writes a CSV file separated by columns.
    :param input_file:
    :param output_file_x:
    :param output_file_y:
    :return:
    '''
    try:
        dir_name_x = os.path.dirname(output_file_x)
        dir_name_y = os.path.dirname(output_file_y)
        os.makedirs(dir_name_x, exist_ok=True)
        os.makedirs(dir_name_y, exist_ok=True)
        header, data = read_csv(input_file)
        x_data = [[row[0]] for row in data]
        y_data = [[t for t in row[1:]] for row in data]
        write_csv(output_file_x, [header[0]], x_data)
        write_csv(output_file_y, [*header[1:]], y_data)
    except Exception as e:
        logging.exception(f"Can't split scv by columns: {e}\n{e.args}\n")


def split_csv_by_years(input_file: str, output_folder: str) -> None:
    '''
    This function writes a CSV file separated by year.
    :param input_file:
    :param output_folder:
    :return:
    '''
    try:
        os.makedirs(output_folder, exist_ok=True)
        header, data = read_csv(input_file)
        data_dict = {}
        for row in data:
            date = datetime.strptime(row[0], '%Y-%m-%d')
            year = date.year
            if year not in data_dict:
                data_dict[year] = []
            data_dict[year].append(row)
        for year, year_data in data_dict.items():
            output_file = f'{output_folder}/{year}0101_{year}1231.csv'
            write_csv(output_file, header, year_data)
    except Exception as e:
        logging.exception(f"Can't split scv by years: {e}\n{e.args}\n")


def split_csv_by_weeks(input_file: str, output_folder: str) -> None:
    '''
    This function writes a CSV file separated by week.
    :param input_file:
    :param output_folder:
    :return:
    '''
    try:
        os.makedirs(output_folder, exist_ok=True)
        header, data = read_csv(input_file)
        data_dict = {}
        for row in data:
            date = datetime.strptime(row[0], '%Y-%m-%d')
            week_start = date - timedelta(days=date.weekday())
            if week_start not in data_dict:
                data_dict[week_start] = []
            data_dict[week_start].append(row)

        for week_start, week_data in data_dict.items():
            week_end = week_start + timedelta(days=6)
            output_file = f'{output_folder}/{week_start.strftime("%Y%m%d")}_{week_end.strftime("%Y%m%d")}.csv'
            write_csv(output_file, header, week_data)
    except Exception as e:
        logging.exception(f"Can't split scv by weeks: {e}\n{e.args}\n")


def read_data_for_date(file_path: str, target_date: datetime) -> dict:
    '''
    This function returns data by date
    :param file_path:
    :param target_date:
    :return:
    '''
    try:
        header, data = read_csv(file_path)
        data_dict = {datetime.strptime(row[0], '%Y-%m-%d'): [t for t in row] for row in data}
        return data_dict.get(target_date)
    except Exception as e:
        logging.exception(f"Can't read data for date: {e}\n{e.args}\n")


class DateIterator:
    def __init__(self, file_path: str):
        header, data = read_csv(file_path)
        self.data_dict = {datetime.strptime(row[0], '%Y-%m-%d'): [t for t in row] for row in data}
        self.dates = sorted(self.data_dict.keys())
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        if self.index < len(self.dates):
            date = self.dates[self.index]
            data = self.data_dict[date]
            self.index += 1
            return date, data
        else:
            raise StopIteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My example explanation')
    parser.add_argument(
        '--mode',
        type=str,
        default='X_Y',
        help='provide a string (mode: "X_Y" or "years" or "weeks" or "find" default: X_Y)'
    )
    parser.add_argument(
        '--path',
        type=str,
        default='test.csv',
        help='provide a string (default: test.csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/',
        help='provide a datetime (default: output/)'
    )
    parser.add_argument(
        '--date',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        default=datetime(2008, 3, 1),
        help='provide a datetime (default: datetime.now())'
    )
    namespace = parser.parse_args()

    if namespace.mode == 'X_Y':
        split_csv_by_columns(input_file=namespace.path,
                             output_file_x=f'{os.path.dirname(namespace.output_dir)}/X.csv',
                             output_file_y=f'{os.path.dirname(namespace.output_dir)}/Y.csv')
    if namespace.mode == 'years':
        split_csv_by_years(input_file=namespace.path,
                           output_folder=f'{os.path.dirname(namespace.output_dir)}/years')
    if namespace.mode == 'weeks':
        split_csv_by_weeks(input_file=namespace.path,
                           output_folder=f'{os.path.dirname(namespace.output_dir)}/weeks')
    if namespace.mode == 'find':
        try:
            r = read_data_for_date(file_path=namespace.path,
                                   target_date=namespace.date)
            if r is not None:
                print(*r)
            else:
                logging.info("Not found!")
        except Exception as e:
            logging.info(f"Empty date {e}/{e.args}")

    if namespace.mode == 'DateIterator':
        iterator = DateIterator('test.csv')
        for date, data in iterator:
            logging.info(date, *data)
