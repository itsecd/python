import os
import requests
import csv
import datetime
import logging
import argparse


logging.basicConfig(level=logging.INFO)


def create_or_clear_csv_file(path: str,
                             filename: str
                             ) -> None:
    """Function take filename and create csv file. If file already created it will be deleted"""
    try:
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, filename).replace('\\', '/'), "w", newline="") as file:
            pass
    except Exception as ex:
        logging.error(f"Couldn't open or clear file: {ex}\n{ex.args}\n")


def write_data_to_csv(filename: str,
                      data: list
                      ) -> None:
    """Function writing data from list to csv file"""
    try:
        with open(filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
    except Exception as ex:
        logging.error(f"Can't write data to file: {ex}\n{ex.args}\n")


def get_json(date: str, 
             url: str = "https://www.cbr-xml-daily.ru/archive/{date}/daily_json.js"
             ) -> dict:
    """Function getting json_file from html page"""
    try:
        date.replace('-', '/')
        url=url.format(date=date)
        html_page = requests.get(url)
        if html_page.status_code != 200:
            return {"error": "Page not found"}
        json_page = html_page.json()
        return json_page
    except Exception as ex:
        logging.error(f"Can't get json file: {ex}\n{ex.args}\n")


def usd_parser():
    """the final function that uses the previous ones and
       parses usd exchange rate from html pages"""
    current_date = datetime.date.today()
    end_date = args.end_date
    while str(current_date).replace('-', '/') != end_date:
        date_str = str(current_date).replace('-', '/')
        json_page = get_json(date_str)
        if "error" in json_page:
            write_data_to_csv(os.path.join(args.path, "data.csv").replace('\\', '/'), [date_str, "Page not found"])
            current_date -= datetime.timedelta(days=1)
            continue
        else:
            usd_value = json_page["Valute"]["USD"]["Value"]
            data = [date_str, str(usd_value)]
            write_data_to_csv(os.path.join(args.path, "data.csv").replace('\\', '/'), data)           
            current_date -= datetime.timedelta(days=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="taking usd exchange rate from html page")
    parser.add_argument('--url',
                        type=str, default="https://www.cbr-xml-daily.ru/archive/{date}/daily_json.js",
                        help='Input url with usd exchange rate'
                        )
    parser.add_argument('--end_date',
                        type=str, default='1998/01/01',
                        help='Input end date for loop'
                        )
    parser.add_argument('--path',
                        type=str, default='csv_files',
                        help='Input path to the file'
                        )
    args = parser.parse_args()
    create_or_clear_csv_file(args.path, 'data.csv')
    usd_parser()