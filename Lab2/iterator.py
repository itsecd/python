import csv
import os
import json


class Iterator:
    """Returns a path to a file belonging of the class"""
    def __init__(self, csv_path: str, class_name: str):
        self.data = list()
        self.count = 0
        self.class_mark = class_name
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if self.class_mark == row[2]:
                    self.data.append(row[0])

    def __iter__(self):
        return self
    
    def __next__(self) -> str:
        if self.count < len(self.data):
            self.count += 1
            return self.data[self.count-1]
        else:
            raise StopIteration


if __name__ == "__main__":
    with open(os.path.join("Lab2", "settings.json"), "r") as settings:
        settings = json.load(settings)
    iter = Iterator(os.path.join(settings["csv_folder"], settings["randomized_csv"]), settings["classes"][0])
    for i in iter:
        print(i)