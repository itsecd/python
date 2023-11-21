import os
import shutil
import csv
import json
import logging
import random
from enum import Enum

logging.basicConfig(level=logging.INFO)


def create_folder(folder_name: str) -> None:
    """The function takes folder name and create folder,  if it does not exist"""
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except Exception as ex:
        logging.error(f"Failed to create folder:{ex.message}\n{ex.args}\n")


class CopyType(Enum):
    RANDOM = "random"
    NUMBERED = "numbered"


def copy_dataset(main_folder: str, new_copy_name: str, copy_type: str = "numbered") -> None:
    """Copy the dataset with annotations to a new directory and create an annotation file.

    main_folder (str): The path to the main folder where the dataset is located.
    new_copy_name (str): The name of the new directory to which the dataset will be copied.
    copy_type (str): Type of data copy. Options: "random" or "numbered".
    """
    create_folder(new_copy_name)

    annotations = []
    existing_random_numbers = set()

    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".jpg"):
                query = os.path.basename(root)
                destination_folder = os.path.join(new_copy_name, query)
                create_folder(destination_folder)

                if copy_type == CopyType.RANDOM:
                    random_number = random.randint(0, 10000)
                    while random_number in existing_random_numbers:
                        random_number = random.randint(0, 10000)

                    existing_random_numbers.add(random_number)

                    new_filename = f"{random_number}.jpg"
                elif copy_type == CopyType.NUMBERED:
                    new_filename = f"{query}_{len(os.listdir(destination_folder)) + 0:04d}.jpg"
                else:
                    logging.error(
                        "Invalid copy_type. Use CopyType.RANDOM or CopyType.NUMBERED.")
                    return

                source_filepath = os.path.join(root, file)
                destination_filepath = os.path.join(
                    destination_folder, new_filename)
                shutil.copyfile(source_filepath, destination_filepath)
                annotations.append(
                    [source_filepath, destination_filepath, query])

    try:
        with open(f"{new_copy_name}.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(annotations)
    except Exception as ex:
        logging.error(f"Failed to write: {ex}\n")


if __name__ == "__main__":
    with open("Lab2/options.json", "r") as options_file:
        options = json.load(options_file)
        copy_dataset(
            options["main_folder"], options["new_copy_name"], CopyType.NUMBERED)
        copy_dataset(
            options["main_folder"], options["random_copy_name"], CopyType.RANDOM)
