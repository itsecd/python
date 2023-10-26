import csv
from next_element import get_path


class ElementIterator:
    """The iterator class gets the path to the cvs file and the class,
    returns the next element if it is of the same class, else it returns None"""

    def __init__(self, name_csv: str, name_class: str) -> None:
        """The method initializes the initial values of the iterator"""
        self.data = list()
        self.count = -1
        with open(name_csv, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                self.data.append(row)
                self.count = 0
        self.limit = len(self.data)
        for i in range(self.limit):
            if self.data[i][2] == name_class:
                self.count = i - 1
                break
        self.mark = name_class
        self.name_csv = name_csv

    def __iter__(self):
        """Returns the iterator itself"""
        return self

    def __next__(self) -> str:
        """Returns the absolute path to the next class element"""
        if self.count < self.limit:
            if self.mark == self.data[self.count + 1][2]:
                self.count += 1
            return self.data[self.count][0]
        else:
            return None


class ClassesIterator(ElementIterator):
    """Iterator by class labels.Accepts the current class,
    returns an object of the next class"""

    def __next__(self) -> str:
        """Returns the absolute path to the next class element"""
        for self.count in range(self.limit):
            if self.mark != self.data[self.count][2]:
                self.mark = self.data[self.count][2]
                tmp = self.data[self.count][0]
                self.data.pop(self.count)
                return tmp
            if len(self.data) == 0:
                return None


if __name__ == "__main__":
    mi = ClassesIterator("Lab2\csv_files\datasets.csv", "rose")
    print(mi.__next__())
    print(mi.__next__())
    print(mi.__next__())
    mi = ElementIterator("Lab2\csv_files\datasets.csv", "rose")
    print(mi.__next__())
    print(mi.__next__())
