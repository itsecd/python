# -*- coding: utf-8 -*-

import logging
import argparse
from datetime import datetime
import os
from division_by_week import split_csv_by_weeks
from division_by_year import split_csv_by_years
from iterator import DateIterator
from spliting_into_two_files import split_csv_by_columns
from get_value_form_date import read_data_for_date

logging.basicConfig(level=logging.INFO)


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
