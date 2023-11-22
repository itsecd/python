from file_manipulation import read_csv
from datetime import datetime


class DateIterator:
    def __init__(self, file_path: str):
        header, data = read_csv(file_path)
        self.data_dict = {datetime.strptime(row[0], '%Y-%m-%d'): [t for t in row] for row in data}
        self.dates = sorted(self.data_dict.keys())
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        if self.index < len(self.dates):
            date = self.dates[self.index]
            data = self.data_dict[date]
            self.index += 1
            return date, data
        else:
            raise StopIteration
