import requests
import csv
import datetime
from datetime import date
from dateutil.rrule import rrule, DAILY

delta = datetime.timedelta(days=30)
end_date = date.today()
start_date = end_date - delta

for d in rrule(DAILY, dtstart=start_date, until=end_date):
    current_day = d.strftime('%d')
    current_month = d.strftime('%m')
    current_year = d.year
    current_date = str(current_year) + "-" + str(current_month) + "-" + str(current_day)
    
    url = 'https://www.cbr-xml-daily.ru/archive/'+str(current_year)+'/'+str(current_month)+'/'+str(current_day)+'/daily_json.js'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
    
        usd_rates = data["Valute"]["USD"]["Value"]

        with open("dataset.csv", "a", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter = ",")

            csv_writer.writerow([current_date, usd_rates])
    else:
         with open("dataset.csv", "a", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter = ",")

            csv_writer.writerow([current_date, "Page not found"])