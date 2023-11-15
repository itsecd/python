import os
import csv
from typing import Generator

def get_next(class_name: str, csv_path: str) -> Generator[str, None, None]:
    """
    This function yields the relative path for each object in the specified directory based on the CSV file.
    """
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        next(reader)  

        for row in reader:
            obj_class, absolute_path = row[2], row[0]
            if class_name == obj_class:
                yield os.path.relpath(absolute_path, 'dataset')

if __name__ == "__main__":
    csv_path = 'annotation.csv'
    cat_generator = get_next('cat', csv_path)

    next_cat_image = next(cat_generator, None)
    while next_cat_image:
        print(next_cat_image)
        next_cat_image = next(cat_generator, None)
