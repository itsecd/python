import os
import shutil
import csv
import random
import logging
from util import write_annotation_to_csv

logging.basicConfig(filename='annotation3.log', level=logging.INFO)


def generate_random_numbers(num_items: int) -> list[int]:
    """
    Generate a list of unique random numbers in the range [0, 10000].
    """
    return random.sample(range(0, 10001), num_items)


def process_images(src_path: str, dest_path: str) -> tuple[list[str], list[str]]:
    """
    Process images by copying, renaming, and returning absolute and relative paths.
    """
    if os.path.isdir(dest_path):
        shutil.rmtree(dest_path)

    old_path = os.path.relpath(src_path)
    new_path = os.path.relpath(dest_path)
    shutil.copytree(old_path, new_path)

    old_names = os.listdir(new_path)
    old_relative_paths = [os.path.join(new_path, name) for name in old_names]

    random_numbers = generate_random_numbers(len(old_names))
    new_names = [f'{number}.jpg' for number in random_numbers]

    for old_name, new_name in zip(old_relative_paths, new_names):
        os.replace(old_name, new_name)

    new_absolute_paths = [os.path.join(os.path.abspath(dest_path), name) for name in new_names]

    return new_absolute_paths, new_names


if __name__ == "__main__":
    dataset_dir = 'dataset3'
    new_absolute_paths, new_names = process_images('dataset2', dataset_dir)

    annotation_data = []
    for absolute_path, relative_path, old_relative_path in zip(
        new_absolute_paths, new_names, [os.path.join(dataset_dir, name) for name in new_names]
    ):
        if 'cat' in old_relative_path:
            name = 'cat'
        else:
            name = 'dog'
        annotation_data.append([absolute_path, relative_path, name])
        logging.info(f"Added entry for {name}: {absolute_path}")

    write_annotation_to_csv('annotation3.csv', annotation_data)
