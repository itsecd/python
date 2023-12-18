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

class ClassIterator:
    def __init__(self, annotation_file: str, class_names: list):
        self.annotation_file = annotation_file
        self.class_names = class_names
        self.current_name_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_name_index < len(self.class_names):
            current_class_name = self.class_names[self.current_name_index]
            self.current_name_index += 1
            return DatasetIterator(self.annotation_file, current_class_name)
        else:
            raise StopIteration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a dataset')
    parser.add_argument('annotation_file', type=str, help='Path to annotation')
    parser.add_argument('class_names', type=str, nargs='+', help='List of class names')

    args = parser.parse_args()

    annotation_file = args.annotation_file
    class_names = args.class_names

    iterator = ClassIterator(annotation_file, class_names)

    for dataset_iterator in iterator:
        for instance in dataset_iterator:
            print(instance)