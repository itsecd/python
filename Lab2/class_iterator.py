"""Module providing a function printing python version 3.11.5."""
import os
import csv

class PhotoIterator:
    def __init__(self, csv_annotation: str, name_class: str):
        self.name_class = name_class
        self.counter = 0
        self.objecs = []
        with open(csv_annotation, "r", newline='') as f:
            reader = csv.DictReader(f)
            for item in reader:
                if item['class'] == name_class:
                    self.objecs.append(item['absolute_path'])
    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.counter < len(self.list):
            print("Заглушка")
        else:
            raise StopIteration("End of iteration")