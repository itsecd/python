# -*- coding: utf-8 -*-

import os
import logging
from datetime import timedelta
from datetime import datetime
from file_manipulation import read_csv
from file_manipulation import write_csv


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
