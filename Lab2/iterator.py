import csv
import logging

logging.basicConfig(filename="py_log1.log", filemode="a", level=logging.INFO)


class Iterator:
    """The iterator class will return
    an element of the same tag"""

    def __init__(self, tag: str, name_csv: str) -> None:
        self.l = []
        self.count = -1
        with open(name_csv, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                self.l.append(row)
                self.count = 0
        self.limit = len(self.l)
        for i in range(self.limit):
            if self.l[i][2] == tag:
                self.count = i - 1
                break
        self.t = tag
        self.csv = name_csv

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.count < self.limit:
            if self.t == self.l[self.count + 1][2]:
                self.count += 1
            return self.l[self.count][0]
        else:
            return None


class TagIterator:
    def __init__(self, path, first_tag, second_tag):
        self.first_it = Iterator(first_tag, path)
        self.second_it = Iterator(second_tag, path)

    def next_first_tag(self):
        return next(self.first_it)

    def next_second_tag(self):
        return next(self.second_it)


if __name__ == "__main__":
    it = TagIterator("file.csv", "tiger", "leopard")
    next_first_tag = it.next_first_tag()
    next_second_tag = it.next_second_tag()
    logging.info(f"Iterator: {next_first_tag},{next_second_tag}")
