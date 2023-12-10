import os
import random
import csv
import json


class ReviewIterator:
    def __init__(self, output_file_: str, class_label: str) -> None:
        self.output_file_ = output_file_
        self.class_label = class_label
        self.instances = self._load_instances()
        self._shuffle_instances()
        self.current_index = 0

    def _load_instances(self) -> list:
        instances = []
        with open(self.output_file_, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                from thefuzz import fuzz as f
                if (f.partial_ratio(row['class_label'], self.class_label) == 100):
                    instances.append(row['absolute_path'])
        return instances

    def _shuffle_instances(self) -> None:
        random.shuffle(self.instances)

    def __iter__(self):
        return self

    def __next__(self) -> None:
        if self.current_index < len(self.instances):
            next_instance = self.instances[self.current_index]
            self.current_index += 1
            return next_instance
        else:
            raise StopIteration


if __name__ == "__main__":
    with open("C://Users/Ceh9/PycharmProjects/pythonProject/Lab2/options.json", "r") as options_file:
        options = json.load(options_file)
    iterator =ReviewIterator(options["output_file_"], options["class_label"][0])
    for iter in iterator:
        print("Следующий экземпляр:", iter)