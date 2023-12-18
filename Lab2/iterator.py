import argparse
import csv

class DatasetIterator:
    def __init__(self, annotation_file: str, class_name: str):
        self.annotation_file = annotation_file
        self.class_name = class_name
        self.class_instances = None
        self.current_index = 0

    def __iter__(self):
        self.class_instances = self._get_class_instances()
        return self

    def _get_class_instances(self):
        instances = []
        with open(self.annotation_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if len(row) == 3 and row[2] == self.class_name:
                    instances.append(row[0])
        return instances

    def __next__(self):
        if self.current_index < len(self.class_instances):
            next_instance = self.class_instances[self.current_index]
            self.current_index += 1
            return next_instance
        else:
            raise StopIteration