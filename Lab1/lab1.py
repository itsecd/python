import argparse
import csv
import os
from datetime import date, timedelta
import requests
import logging


logging.basicConfig(level=logging.INFO)


def create_folder(base_folder: str = "dataset") -> None:
    """The function form a folder"""
    try:
        if not os.path.exists(base_folder):
            os.mkdir(base_folder)
    except Exception as ex:
        logging.exception(f"Can't create folder: {ex.message}\n{ex.args}\n")


def make_lists(start_date: date,
               end_date: date,
               delta: timedelta,
               url_template: str = 'https://www.cbr-xml-daily.ru/archive/{year}/{month}/{day}/daily_json.js'
               ) -> list:
    """The function takes start date, end date, delta,
    passes through all dates with a given delta
    and takes the dollar rate from the site.
    """
    list_usd = []
    list_dates = []
    try:
        while (start_date != end_date):
            current_day = start_date.strftime('%d')
            current_month = start_date.strftime('%m')
            current_year = start_date.year
            url = url_template.format(year=current_year, month=current_month, day=current_day)
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                usd_rates = data["Valute"]["USD"]["Value"]
                list_usd.append(usd_rates)
                list_dates.append(str(start_date))
            start_date -= delta
        return list_dates, list_usd
    except Exception as ex:
        logging.error(f"can't get data: {ex.message}\n{ex.args}\n")


def write_to_f(filename: str,
               dates: list,
               data: list,
               file_path: str
               ) -> None:
    """The function takes filename and two lists of data,
    then writes data to a file with the specified name.
    """
    try:
        output_file_path = os.path.join(file_path, filename).replace("\\","/")
        create_folder(file_path)
        with open(output_file_path, "a", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            combined_list = [(date, usd) for date, usd in zip(dates, data)]
            csv_writer.writerows(combined_list)
    except Exception as ex:
        logging.error(f"Can't write data to file: {ex}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect USD exchange rates.')
    parser.add_argument('--delta',
                        type=int, default=1,
                        help='Step in while'
                        )
    parser.add_argument('--delta2',
                        type=int, default=9000,
                        help='Number of days'
                        )
    parser.add_argument('--output',
                        type=str, default='dataset.csv',
                        help='Output file name'
                        )
    parser.add_argument('--path_file',
                        type=str, default='dataset',
                        help='The path to the data file'
                        )
    parser.add_argument('--url_template',
                        type=str, default='https://www.cbr-xml-daily.ru/archive/{year}/{month}/{day}/daily_json.js',
                        help='URL template for fetching data'
                        )
    
    args = parser.parse_args()

    delta = timedelta(days=args.delta)
    delta2 = timedelta(days=args.delta2)
    start_date = date.today()
    end_date = start_date - delta2

    dates, result = make_lists(start_date, end_date, delta, args.url_template)
    write_to_f(args.output, dates, result, args.path_file)