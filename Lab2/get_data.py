import argparse
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Generator, Tuple, Union


logging.basicConfig(level=logging.INFO)


def read_data_from_original_file(date: datetime,
                           file_path: str
                           ) -> str:
    """the function takes the date for which the data needs to be found,
    the path to the files, and returns the data"""

    data = None
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            date_to_compare = date.date()

            if date_to_compare in df['Date'].dt.date.values:
                value = df.loc[df['Date'].dt.date == date_to_compare, 'Value'].values[0]
                if value != "data not found":
                    data = value

        return data
    except Exception as ex:
        logging.exception(f"Can't read data from original file: {ex}\n{ex.args}\n")


def read_data_from_x_y(date: datetime,
                       dates_file_path: str,
                       values_file_path: str
                       ) -> str:
    """the function takes the date for which the data needs to be found,
    the path to the files x and y, and returns the data"""

    data = None
    try:
        if os.path.exists(dates_file_path) and os.path.exists(values_file_path):
            dates_df = pd.read_csv(dates_file_path)
            values_df = pd.read_csv(values_file_path)
            
            dates_df['Date'] = pd.to_datetime(dates_df['Date'], format='%Y/%m/%d')
            date_to_compare = date.date()
            dates_filtered = dates_df[dates_df['Date'].dt.date == date_to_compare]

            if not dates_filtered.empty:
                index = dates_filtered.index[0]
                value = values_df.iloc[index]['Value']
                if value != "data not found":
                    data = value

        return data
    except Exception as ex:
        logging.exception(f"Can't read data from x and y files: {ex}\n{ex.args}\n")


def read_data_from_years(date: datetime,
                        folder_path: str
                        ) -> str:
    """the function takes the date for which the data needs to be found,
    the path to the files divided by years, and returns the data"""

    data = None
    try:
        if os.path.exists(folder_path):
            matching_file = None
            date_to_find = date.date()
            
            for file in os.listdir(folder_path):
                file_date_parts = file.split('_')
                if len(file_date_parts) == 2:
                    start_date_str, end_date_str = file_date_parts[0], file_date_parts[1].split('.')[0]
                    start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
                    end_date = datetime.strptime(end_date_str, "%Y%m%d").date()
                    if start_date <= date_to_find <= end_date:
                        matching_file = file
                        break

            if matching_file:
                file_path = os.path.join(folder_path, matching_file)
                df = pd.read_csv(file_path)

                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
                filtered_data = df[df['Date'].dt.date == date_to_find]

                if not filtered_data.empty:
                    value = filtered_data.iloc[0]['Value']
                    if value != "data not found":
                        data = value

        return data
    except Exception as ex:
        logging.exception(f"Can't read data from files, divided by years: {ex}\n{ex.args}\n")


def read_data_from_weeks(date: datetime,
                        folder_path: str,
                        ) -> str:
    """the function takes the date for which the data needs to be found,
    the path to the files divided by weeks, and returns the data"""
    data = None
    try:
        if os.path.exists(folder_path):
            matching_file = None
            date_to_find = date.date()
            
            for file in os.listdir(folder_path):
                file_date_parts = file.split('_')
                if len(file_date_parts) == 2:
                    start_date_str, end_date_str = file_date_parts[0], file_date_parts[1].split('.')[0]
                    start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
                    end_date = datetime.strptime(end_date_str, "%Y%m%d").date()
                    if start_date <= date_to_find <= end_date:
                        matching_file = file
                        break

            if matching_file:
                file_path = os.path.join(folder_path, matching_file)
                df = pd.read_csv(file_path)

                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
                filtered_data = df[df['Date'].dt.date == date_to_find]

                if not filtered_data.empty:
                    value = filtered_data.iloc[0]['Value']
                    if value != "data not found":
                        data = value

        return data
    except Exception as ex:
        logging.exception(f"Can't read data from files, divided by weeks: {ex}\n{ex.args}\n")


def get_data_for_date(date: datetime,
                      input_csv:str,
                      dates_file_path: str,
                      values_file_path: str
                      ) -> str:
    """the function tries to use 4 different methods to get the data"""
    data = None
    try:
        for folder_path in folder_paths:
            if folder_path == 'csv_files':
                data = read_data_from_original_file(date, input_csv)
            elif folder_path == 'X_and_Y':
                data = read_data_from_x_y(date, dates_file_path,values_file_path)
            elif folder_path == 'years':
                data = read_data_from_years(date, folder_path)
            elif folder_path == 'weeks':
                data = read_data_from_weeks(date, folder_path)

            if data is not None:
                break

        return data
    except Exception as ex:
        logging.exception(f"Can't read data from files: {ex}\n{ex.args}\n")


def next_date(start_date: datetime,
              end_date: datetime,
              input_csv:str,
              dates_file_path: str,
              values_file_path: str
              ) -> Generator[Tuple[datetime, Union[str, None]], None, None]:
    """the function move date to the next"""
    def get_next_valid_date(start_date):
        while start_date <= end_date:
            data = get_data_for_date(start_date,input_csv,dates_file_path,values_file_path)
            start_date += timedelta(days=1)
            if data is not None and data != "data not found":
                return start_date - timedelta(days=1), data
        return None, None

    while start_date <= end_date:
        date, data = get_next_valid_date(start_date)
        if date is None:
            break
        yield date, data
        start_date = date + timedelta(days=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split csv file for weeks.')
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "../Lab1/dataset/dataset.csv")
    folder_paths = [
    input_csv,
    'X_and_Y',
    'years',
    'weeks'
    ] 
    data_for_date = get_data_for_date(args.date_to_find,input_csv, args.output_x, args.output_y)
    print(f"Value for: {args.date_to_find}: {data_for_date}")

    data_iterator = next_date(args.start_date,args.end_date,input_csv, args.output_x, args.output_y)

    for _ in range(5):
        next_date, next_data_value = next(data_iterator)
        if next_date is not None:
            print(f"Date: {next_date}, Value: {next_data_value}")
        else:
            print("Больше данных нет.")
            break