import os
import csv
from pathlib import Path
import logging


logging.basicConfig(filename='annotation.log', level=logging.INFO)


def get_paths(name: str) -> tuple[list[Path], list[Path]]:
    """
    This function returns a tuple of absolute and relative paths for all images 
    of the specific name of the animal passed to the function.
    """
    absolute_path = os.path.abspath(os.path.join('dataset', name))
    image_paths = [os.path.join(absolute_path, img) for img in os.listdir(absolute_path)]

    relative_path = os.path.relpath(os.path.join('dataset', name))
    relative_paths = [os.path.join(relative_path, img) for img in os.listdir(relative_path)]

    return image_paths, relative_paths

def main() -> None:
    cat, dog = 'cat', 'dog'

    cat_absolute_paths, cat_relative_paths = get_paths(cat)
    dog_absolute_paths, dog_relative_paths = get_paths(dog)

    with open('annotation.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for absolute_path, relative_path, label in zip(cat_absolute_paths, cat_relative_paths, [cat] * len(cat_absolute_paths)):
            writer.writerow([absolute_path, relative_path, label])
            logging.info(f"Added entry for {label}: {absolute_path}")

        for absolute_path, relative_path, label in zip(dog_absolute_paths, dog_relative_paths, [dog] * len(dog_absolute_paths)):
            writer.writerow([absolute_path, relative_path, label])
            logging.info(f"Added entry for {label}: {absolute_path}")

if __name__ == "__main__":
    main()
