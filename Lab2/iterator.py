import csv
import os
from itertools import cycle


class ImageIterator:
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file
        self.class_instances = self.load_annotation_file()
        self.class_cycle = cycle(self.class_instances)
        self.prev_instances = set()

    def load_annotation_file(self):
        instances = []
        with open(self.annotation_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if len(row) == 3:
                    instances.append(row[1])
        return instances

    def get_next_instance(self, target_class):
        for instance in self.class_cycle:
            _, instance_name = os.path.split(instance)
            parts = instance_name.split("_")
            if len(parts) >= 2:
                instance_class = parts[0]
                instance_class = instance_class.split(".")[0]
                if instance_class == target_class and instance not in self.prev_instances:
                    self.prev_instances.add(instance)
                    return instance
        return None


if __name__ == "__main__":
    annotation_file = 'copy_dataset.csv'
    target_class = 'leopard'

    image_iterator = ImageIterator(annotation_file)

    while True:
        next_instance = image_iterator.get_next_instance(target_class)
        if next_instance is not None:
            print(next_instance)
        else:
            print("No more instances for the specified class.")
            break
