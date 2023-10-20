import csv

class Iterator:
    def __init__(self, path_csv : str, object : str):
        self.path_csv = path_csv
        self.object = object
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
            if self.object == self.list[self.counter][2]:
                return self.list[self.counter][1]
        else:
            raise StopIteration
