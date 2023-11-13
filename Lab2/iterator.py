import csv
import os
import json


class PathIterator:
    """This Iterator returns path of the class. When class ends, raises StopIteration"""
    def __init__(self, csv_path: str, name_class: str):
        self.data = list()
        self.count = 0
        self.mark = name_class
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if self.mark == row[2]:
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
    iter = PathIterator(f"{settings["csv"]}/{settings["copy"]}.csv", f"{settings["classes"][0]}")
    for i in iter:
        print(i)