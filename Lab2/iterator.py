import argparse
import csv

class DatasetIterator:
    def __init__(self, annotation_file: str, class_name: str):
        self.annotation_file = annotation_file
        self.class_name = class_name
        self.instances = self._get_class_instances()
        self.current_index = 0

    def _get_class_instances(self):
        instances = []
        with open(self.annotation_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if len(row) == 3 and row[2] == self.class_name:
                    instances.append(row[0])
        return instances

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index < len(self.instances):
            next_instance = self.instances[self.current_index]
            self.current_index += 1
            return next_instance
        else:
            raise StopIteration

class MultiClassDatasetIterator:
    def __init__(self, annotation_file: str, class_names: list):
        self.annotation_file = annotation_file
        self.class_names = class_names
        self.dataset_iterators = [DatasetIterator(annotation_file, class_name) for class_name in class_names]
        self.current_dataset_iterator = None

    def __iter__(self):
        self.current_dataset_iterator = iter(self.dataset_iterators)
        return self

    def __next__(self):
        while self.current_dataset_iterator:
            try:
                return next(self.current_dataset_iterator)
            except StopIteration:
                self.current_dataset_iterator = next(iter(self.dataset_iterators), None)

        raise StopIteration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a dataset')
    parser.add_argument('annotation_file', type=str, help='Path to annotation')
    parser.add_argument('class_names', type=str, nargs='+', help='List of class names')

    args = parser.parse_args()

    annotation_file = args.annotation_file
    class_names = args.class_names

    multi_class_iterator = MultiClassDatasetIterator(annotation_file, class_names)

    for instance in multi_class_iterator:
        print(instance)