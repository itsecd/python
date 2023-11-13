import csv
import argparse
from path_of_next import get_path_of_next


class RevIterator:
    def __init__(self, path_to_csv: str, class_label: int) -> None:
        self.class_label = class_label
        self.path_to_csv = path_to_csv
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self) -> str:
        self.counter += 1
        try:
            return get_path_of_next(self.class_label, self.counter - 1, self.path_to_csv)
        except:
            raise StopIteration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input csv path, label of class")
    parser.add_argument("-c", "--csv", help="Input csv path", type=str)
    parser.add_argument("-l", "--label", help="Input label of class", type=int)
    args = parser.parse_args()
    iter = RevIterator(args.csv, args.label)
    for i in iter:
        print(i)
