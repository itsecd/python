import csv
import random


def next_instance(class_label: str) -> str:
    instances = []

    with open('annotation.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            _, _, current_class = row
            if current_class == class_label:
                instances.append(row)

    random.shuffle(instances)

    for instance in instances:
        return instance[0]

if __name__ == "__main__":
    class_label = 'polar_bear'
    next_polar_bear_instance = next_instance(class_label)

    if next_polar_bear_instance:
        print(f"Next instance for class {class_label}: {next_polar_bear_instance}")
    else:
        print(f"No more instances for class {class_label}")
