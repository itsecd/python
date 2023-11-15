import csv
from pathlib import Path
import logging

logging.basicConfig(filename='annotation.log', level=logging.INFO)


def get_absolute_paths(name: str) -> list[Path]:
    absolute_path = Path('dataset') / name
    return list(absolute_path.glob('*'))


def get_relative_paths(name: str) -> list[Path]:
    relative_path = Path('dataset') / name
    return list(relative_path.glob('*'))


def write_annotation_to_csv(csv_writer, absolute_paths, relative_paths, label):
    for absolute_path, relative_path in zip(absolute_paths, relative_paths):
        csv_writer.writerow([str(absolute_path), str(relative_path), label])
        logging.info(f"Added entry for {label}: {absolute_path}")


if __name__ == "__main__":
    cat, dog = 'cat', 'dog'

    cat_absolute_paths = get_absolute_paths(cat)
    cat_relative_paths = get_relative_paths(cat)

    dog_absolute_paths = get_absolute_paths(dog)
    dog_relative_paths = get_relative_paths(dog)

    with open('annotation.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        write_annotation_to_csv(writer, cat_absolute_paths, cat_relative_paths, cat)
        write_annotation_to_csv(writer, dog_absolute_paths, dog_relative_paths, dog)
