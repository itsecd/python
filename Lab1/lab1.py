import csv
from datetime import date, timedelta
import requests

def make_lists(start_date: date,
               end_date: date,
               delta: timedelta
               ) -> list:
    list_usd = []
    list_dates = []
    while (start_date!=end_date):
        current_day = start_date.strftime('%d')
        current_month = start_date.strftime('%m')
        current_year = start_date.year
        url: str = f'https://www.cbr-xml-daily.ru/archive/{current_year}/{current_month}/{current_day}/daily_json.js'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            usd_rates = data["Valute"]["USD"]["Value"]
            list_usd.append(usd_rates)
            list_dates.append(str(start_date))
        start_date += delta
    return list_dates, list_usd

def write_to_f(filename: str,
               dates: list,
               data: list
               ) -> None:
    with open(filename, "a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file,delimiter=",")
        combined_list = [(date,usd) for date, usd in zip(dates,data)]
        csv_writer.writerows(combined_list)

delta = timedelta(days=1)
delta2 = timedelta(days=30)
end_date = date.today()
start_date = end_date - delta2

dates, result = make_lists(start_date, end_date, delta)
write_to_f("dataset.csv", dates,result)