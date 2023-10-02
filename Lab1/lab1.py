import requests
import csv
import datetime
from datetime import date

delta = datetime.timedelta(days=30)
end_date = date.today()
start_date = end_date - delta

url = 'https://www.cbr-xml-daily.ru/archive/'+str(start_date).replace("-","/")+'/daily_json.js'
response = requests.get(url)

if response.status_code == 200:
        data = response.json()
    
        usd_rates = data["Valute"]["USD"]["Value"]

        with open("dataset.csv", "a", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter = ",")

            csv_writer.writerow([start_date, usd_rates])