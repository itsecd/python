import argparse
import os
import logging
import csv
from datetime import datetime
from create_folder import create_folder


logging.basicConfig(level=logging.INFO)


def split_by_year(input_file: str,
                  output_path: str
                  ) -> None:
    """The function takes path to the input file and split file to years"""
    try:
        create_folder(output_path)

        with open(input_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            date_index = header.index('Date')
            value_index = header.index('Value')

            year_data = {}

            for row in reader:
                date_str = row[date_index]
                value = row[value_index]

                date = datetime.strptime(date_str, '%Y-%m-%d')
                year = date.year

                if year not in year_data:
                    year_data[year] = []

                year_data[year].append([date_str, value])

            for year, data in year_data.items():

                output_file = os.path.join(output_path, f"{year}0101_{year}1231.csv")

                with open(output_file, 'w', newline='') as output_csv:
                    writer = csv.writer(output_csv)
                    writer.writerow(['Date', 'Value'])
                    writer.writerows(data)
    except Exception as ex:
        logging.exception(f"Can't split data to years: {ex}\n{ex.args}\n")


def read_data_from_years(date: datetime,
                         folder_path: str
                         ) -> str:
    """The function takes the date for which the data needs to be found,
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
                with open(file_path, 'r') as file:
                    reader = csv.reader(file)
                    header = next(reader)  # Read header
                    date_index = header.index('Date')
                    value_index = header.index('Value')

                    for row in reader:
                        row_date = datetime.strptime(row[date_index], '%Y-%m-%d').date()
                        if row_date == date_to_find:
                            value = row[value_index]
                            if value != "data not found":
                                data = value
                                break
        return data
    except Exception as ex:
        logging.exception(f"Can't read data from files, divided by years: {ex}\n{ex.args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split csv file for years.')
    parser.add_argument('--path_file',
                        type=str, default='years',
                        help='The path to the data file'
                        )
    
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "../Lab1/dataset/dataset.csv")

    split_by_year(input_csv,args.path_file)