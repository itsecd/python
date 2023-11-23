# -*- coding: utf-8 -*-

import os
import logging
from datetime import datetime
from file_manipulation import read_csv
from file_manipulation import write_csv


def split_csv_by_years(input_file: str, output_folder: str) -> None:
    """
    This function writes a CSV file separated by year.
    :param input_file:
    :param output_folder:
    :return:
    """
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
