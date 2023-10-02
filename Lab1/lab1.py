import os
import requests
import csv
import datetime



start_date = datetime.date(2022,2,2)
URL = "https://www.cbr-xml-daily.ru/archive/"+str(start_date).replace("-","/")+"/daily_json.js"
html_page = requests.get(URL)
json_page = html_page.json()
usd_value = json_page["Valute"]["USD"]["Value"]
data = [str(start_date)], [str(usd_value)]
with open("data.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(data)
print(str(start_date)+", "+str(usd_value))