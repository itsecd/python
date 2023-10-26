"""Module providing a function printing python version 3.11.5."""
import os
import csv
import json
import logging


logging.basicConfig(level=logging.DEBUG)


class PhotoIterator:
    """
    The iterator is intended for only one class, 
    it goes through all the links in the annotation until they run out
    """
    def __init__(self, csv_annotation: str, name_class: str):
        self.name_class = name_class
        self.counter = 0
        self.objects = []
        with open(csv_annotation, "r", newline='') as f:
            reader = csv.DictReader(f)
            for item in reader:
                if item['class'] == name_class:
                    self.objects.append(item['absolute_path'])

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.counter < len(self.objects):
            self.counter += 1
            return self.objects[self.counter - 1]
        else:
            raise StopIteration("End of iteration")


if __name__ == "__main__":
    with open(os.path.join("Lab2","user_settings.json"), "r", newline='') as f:
        settings = json.load(f)
    ph_iter = PhotoIterator(settings["annotation_for_iterating"], settings["class"])
    try:
        while True:
            logging.info(next(ph_iter))
    except StopIteration as e:
        logging.debug(e)
