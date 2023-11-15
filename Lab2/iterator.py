import shutil
import os
import csv
import json


class DatasetIterator:
    def __init__(self, dataset_path: str, class_name: str):
        self.data = list()
        self.count = 0
        self.mark = class_name
        with open(dataset_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if self.mark == row[2]:
                    self.data.append(row[0])

    def __iter__(self):
        return self

    def __next__(self) -> str:
        def __next__(self) -> str:
            if self.count < len(self.data):
                self.count += 1
                return self.data[self.count-1]
            else:
                raise StopIteration


if __name__ == "__main__":
    with open(os.path.join("Lab2", "settings.json"), "r") as settings:
        settings = json.load(settings)
    iter = DatasetIterator(os.path.join(settings["directory"], settings["randomized_csv"]), settings["classes"][0])
    for i in iter:
        print(i)
       
