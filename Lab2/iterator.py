import csv
import os
import json

class ImgIterator:
    def __init__(self, path_csv : str, name_class : str):
        self.path_csv = path_csv
        self.list = []
        self.mark = name_class

        self.counter = 0

        with open(self.path_csv, 'r') as file:
            csv_file = csv.reader(file)
            for row in csv_file:
                if self.mark == row[2]:
                    self.list.append(row)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.counter < len(self.list):
            self.counter += 1
            return self.list[self.counter-1][0]
        else:
            raise StopIteration
        

# Testing ImgIterator
if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "json", "setting.json"), 'r') as setting_json:
        setting = json.load(setting_json)

    iter = ImgIterator(f'{os.path.join(setting["name-dir"], setting["name-csv-dir"], setting["name-normal"])}.csv', 
                       'tulip')

    for i in iter:
        print(i)

