import csv
import os

class ImageIterator:
    def __init__(self, annotation_file, target_class, dest_folder_path=None):
        self.annotation_file = annotation_file
        self.target_class = target_class
        self.dest_folder_path = dest_folder_path
        self.class_instances = self.load_annotation_file()
        self.index = 0
        self.prev_instances = set()

    def load_annotation_file(self):
        instances = []
        with open(self.annotation_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if len(row) == 3 and row[2] == self.target_class:
                    instances.append(row[0])
        return instances

    def __iter__(self):
        return self

    def __next__(self):
        while self.index < len(self.class_instances):
            instance = self.class_instances[self.index]
            self.index += 1
            if instance not in self.prev_instances:
                self.prev_instances.add(instance)
                return instance

        raise StopIteration("No more instances for the specified class.")


class ClassIterator:
    def __init__(self, csv_file: str, class_labels, dest_folder_path):
        self.csv_file = csv_file
        self.class_labels = class_labels
        self.dest_folder_path = dest_folder_path
        self.image_iterators = {label: ImageIterator(csv_file, label, dest_folder_path) for label in class_labels}
        self.current_label_index = 0
        self.current_image_iterator = None


    def __iter__(self):
        return self

    def __next__(self):
        if not self.class_labels or self.current_label_index >= len(self.class_labels):
            raise StopIteration("No more instances for the specified class.")

        current_class_label = self.class_labels[self.current_label_index]

        if self.current_image_iterator is None:
            self.current_image_iterator = self.image_iterators[current_class_label]

        try:
            return next(self.current_image_iterator)
        except StopIteration:
            self.current_label_index += 1
            self.current_image_iterator = None
            return next(self)


if __name__ == "__main__":
    annotation_file = 'annotation.csv'
    class_labels = ["zebra", "bayhorse"]
    iterator = ClassIterator(annotation_file, class_labels)

    instance = iterator.next_image()
    while instance is not None:
        print(instance)
        instance = iterator.next_image()