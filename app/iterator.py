import csv
import os
import json

class CsvIterator:
    def __init__(self, path_csv : str):
        self.path_csv = path_csv
        self.list = []

        self.counter = 0

        with open(self.path_csv, 'r') as file:
            csv_file = csv.reader(file)
            for row in csv_file:
                self.list.append(row)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.counter < len(self.list):
            self.counter += 1
            return self.list[self.counter-1]
        else:
            raise StopIteration
        

# Testing CsvIterator
if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "json", "src.json"), 'r') as src_json:
        src = json.load(src_json)

    with open(os.path.join(os.path.dirname(__file__), "json", "name.json"), 'r') as name_json:
        name = json.load(name_json)

    iter = CsvIterator(f'{os.path.join(src["name-dir"], src["name-csv-dir"], name["name-normal"])}.csv')

    for i in iter:
        print(f'{i[0]} - {i[1]} - {i[2]}')

