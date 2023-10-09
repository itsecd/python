import argparse
import csv
from datetime import date, timedelta
import requests
import logging


logging.basicConfig(level=logging.INFO)


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
               data: list
               ) -> None:
    """The function takes filename and two lists of data,
    then writes data to a file with the specified name.
    """
    try:
        with open(filename, "a", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            combined_list = [(date, usd) for date, usd in zip(dates, data)]
            csv_writer.writerows(combined_list)
    except Exception as ex:
        logging.error(f"Can't write data to file: {ex.message}\n{ex.args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect USD exchange rates.')
    parser.add_argument('--delta',
                        type=int, default=1,
                        help='Delta in days'
                        )
    parser.add_argument('--delta2',
                        type=int, default=9000,
                        help='Delta2 in days'
                        )
    parser.add_argument('--output',
                        type=str, default='dataset.csv',
                        help='Output file path'
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
    write_to_f(args.output, dates, result)