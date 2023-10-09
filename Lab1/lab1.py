import csv
from datetime import date, timedelta
import requests
import logging


logging.basicConfig(level=logging.INFO)


def make_lists(start_date: date,
               end_date: date,
               delta: timedelta
               ) -> list:
    """The function takes start date, end date, delta,
    passes through all dates with a given delta
    and takes the dollar rate from the site.
    """
    list_usd = []
    list_dates = []
    try:
        while (start_date!=end_date):
            current_day = start_date.strftime('%d')
            current_month = start_date.strftime('%m')
            current_year = start_date.year
            url = f'https://www.cbr-xml-daily.ru/archive/{current_year}/{current_month}/{current_day}/daily_json.js'
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
            csv_writer = csv.writer(csv_file,delimiter=",")
            combined_list = [(date,usd) for date, usd in zip(dates,data)]
            csv_writer.writerows(combined_list)
    except Exception as ex:
        logging.error(f"Can't write data to file: {ex.message}\n{ex.args}\n")

delta = timedelta(days=1)
delta2 = timedelta(days=9000)
start_date = date.today()
end_date = start_date - delta2

dates, result = make_lists(start_date, end_date, delta)
write_to_f("dataset.csv", dates,result)