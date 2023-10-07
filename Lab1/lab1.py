import requests
import csv
import datetime
from datetime import date, timedelta
from dateutil.rrule import rrule, DAILY

delta = datetime.timedelta(days=30)
end_date = date.today()
start_date = end_date - delta

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


#with open("dataset.csv", "a", newline="", encoding="utf-8") as csv_file:
#    csv_writer = csv.writer(csv_file, delimiter = ",")
#
#    csv_writer.writerow([current_date, usd_rates])
   