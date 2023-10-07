import os
import requests
import csv
import datetime

def create_or_clear_csv_file(filename):
    with open(filename, "w", newline="") as file:
        pass

def write_data_to_csv(filename, data):
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)

def get_json(date):
    URL = "https://www.cbr-xml-daily.ru/archive/" + date.replace("-", "/") + "/daily_json.js"
    html_page = requests.get(URL)
    if html_page.status_code != 200:
        return {"error": "Page not found"}
    json_page = html_page.json()
    return json_page

def main():
    current_date = datetime.date(2022, 2, 2)
    end_date = datetime.date(2022, 2, 25)
    while current_date != end_date:
        date_str = str(current_date)
        json_page = get_json(date_str)
        if "error" in json_page:
            write_data_to_csv("data.csv", [date_str, "Page not found"])
            current_date += datetime.timedelta(days=1)
            continue
        else:
            usd_value = json_page["Valute"]["USD"]["Value"]
            data = [date_str, str(usd_value)]
            write_data_to_csv("data.csv", data)
            current_date += datetime.timedelta(days=1)

if __name__ == "__main__":
    create_or_clear_csv_file("data.csv")
    main()