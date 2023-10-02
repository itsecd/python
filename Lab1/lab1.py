import requests
import csv
import datetime
from datetime import date

delta = datetime.timedelta(days=30)
end_date = date.today()
start_date = end_date - delta

url = 'https://www.cbr-xml-daily.ru/archive/'+str(start_date).replace("-","/")+'/daily_json.js'
response = requests.get(url)

print(url)