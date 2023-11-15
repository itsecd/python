import argparse
import os
import logging
import csv
from datetime import datetime, timedelta
from create_folder import create_folder


logging.basicConfig(level=logging.INFO)


def split_by_week(input_file: str,
                  output_path: str
                  ) -> None:
    """The function takes path to the input file and split file to weeks"""
    try:
        create_folder(output_path)

        with open(input_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Read header
            date_index = header.index('Date')
            value_index = header.index('Value')

            data_by_week = {}

            for row in reader:
                date_str = row[date_index]
                value = row[value_index]

                date = datetime.strptime(date_str, '%Y-%m-%d')
                week_start = date - timedelta(days=date.weekday())  # Start of the week
                week_end = week_start + timedelta(days=6)  # End of the week

                week_key = (week_start, week_end)

                if week_key not in data_by_week:
                    data_by_week[week_key] = []

                data_by_week[week_key].append([date_str, value])

            for (start_date, end_date), data in data_by_week.items():

                output_file = os.path.join(output_path, f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")

                with open(output_file, 'w', newline='') as output_csv:
                    writer = csv.writer(output_csv)
                    writer.writerow(['Date', 'Value'])  # Write header
                    writer.writerows(data)
    except Exception as ex:
        logging.exception(f"Can't split data to weeks: {ex}\n{ex.args}\n")


def read_data_from_weeks(date: datetime,
                         folder_path: str
                         ) -> str:
    """The function takes the date for which the data needs to be found,
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
        logging.exception(f"Can't read data from files, divided by weeks: {ex}\n{ex.args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split csv file for weeks.')
    parser.add_argument('--path_file',
                        type=str, default='weeks',
                        help='The path to the data file'
                        )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "../Lab1/dataset/dataset.csv")

    split_by_week(input_csv,args.path_file)