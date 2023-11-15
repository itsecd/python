import argparse
import os
import logging
import csv
import re
from datetime import datetime
from X_Y_csv import read_data_from_x_y
from year_csv import read_data_from_years
from week_csv import read_data_from_weeks


logging.basicConfig(level=logging.INFO)


script_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(script_dir, "../Lab1/dataset/dataset.csv")
folder_paths = [
input_csv,
'X_and_Y',
'years',
'weeks'
] 

def read_data_from_original_file(date: datetime,
                                 file_path: str
                                 ) -> str:
    """The function takes the date for which the data needs to be found,
    the path to the files, and returns the data"""
    data = None
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)  # Read header
                date_index = header.index('Date')
                value_index = header.index('Value')

                date_to_compare = date.date()

                for row in reader:
                    row_date = datetime.strptime(row[date_index], '%Y-%m-%d').date()
                    if row_date == date_to_compare:
                        value = row[value_index]
                        if value != "data not found":
                            data = value
                            break
        return data
    except Exception as ex:
        logging.exception(f"Can't read data from original file: {ex}\n{ex.args}\n")


def get_data_for_date(date: datetime,
                      input_csv: str,
                      dates_file_path: str,
                      values_file_path: str,
                      ) -> str:
    """The function tries to use 4 different methods to get the data."""
    data = None
    try:
        for folder_path in folder_paths:
            if re.match(r"dataset\.csv$", os.path.basename(input_csv)):
                data = read_data_from_original_file(date, input_csv)
            elif re.match(r"[XY]\.csv$", os.path.basename(dates_file_path)) and re.match(r"[XY]\.csv$", os.path.basename(values_file_path)):
                data = read_data_from_x_y(date, dates_file_path, values_file_path)
            elif re.match(r"\d{8}_\d{8}\.csv$", os.path.basename(dates_file_path)) and re.match(r"\d{8}_\d{8}\.csv$", os.path.basename(values_file_path)):
                data = read_data_from_years(date, folder_path)
            elif re.match(r"\d{8}_\d{8}\.csv$", os.path.basename(dates_file_path)) and re.match(r"\d{8}_\d{8}\.csv$", os.path.basename(values_file_path)):
                data = read_data_from_weeks(date, folder_path)

        return data

    except Exception as ex:
        logging.exception(f"Can't read data from files: {ex}\n{ex.args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='read data from files.')
    parser.add_argument('--start_date',
                        type=datetime, default=datetime(1998, 1, 2),
                        help='Start date in the files'
                        )
    parser.add_argument('--end_date',
                        type=datetime, default=datetime(2023, 11, 14),
                        help='end date in the files'
                        )
    parser.add_argument('--date_to_find',
                        type=datetime, default=datetime(2023, 8, 22),
                        help='the date to be found'
                        )
    parser.add_argument('--output_x',
                        type=str, default='X.csv',
                        help='Output file name X'
                        )
    parser.add_argument('--output_y',
                        type=str, default='Y.csv',
                        help='Output file name Y'
                        )
    
    args = parser.parse_args()

    data_for_date = get_data_for_date(args.date_to_find,input_csv, args.output_x, args.output_y)
    print(f"Value for: {args.date_to_find}: {data_for_date}")