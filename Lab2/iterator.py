import pandas as pd


class Iterator:
    """The iterator class gets the path to the cvs file and the class,
    returns the next element if it is of the same class, else it returns None"""

    def __init__(self, name_csv: str, name_class: str) -> None:
        """The method initializes the initial values of the iterator"""
        data = pd.read_csv(name_csv)
        self.limit = len(data)
        for i in range(self.limit):
            if data.values[i][2] == name_class:
                self.count = i - 1
                break
        self.counter = -1
        self.mark = name_class
        self.name_csv = name_csv
        self.limit = len(data)

    def __iter__(self):
        """Returns the iterator itself"""
        return self

    def __next__(self) -> str:
        """Returns the absolute path to the next class element"""
        if self.counter < self.limit:
            data = pd.read_csv(self.name_csv)
            if self.mark == data.values[self.counter + 1][2]:
                self.counter += 1
            return data.values[self.counter][0]
        else:
            return None
