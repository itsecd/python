import os
import json
import random
import shutil
import logging
import csv_annotation

logging.basicConfig(level=logging.INFO)


def copy_with_random(
    old_directory: str, classes: list, new_directory: str, name_csv: str
) -> None:
    """The function copies images from class folders to the dataset.
    New file name=random number"""
    try:
        img_list = list()
        numbers = set()
        count_files = len(os.listdir(os.path.join(old_directory, classes[0])))- len(classes)
        while len(numbers) <= (count_files * len(classes)):
            numbers.add(random.randint(0, 10000))
        number = list(numbers)
        for c in classes:
            for i in range(count_files):
                j = len(os.listdir(os.path.join(new_directory))) - len(classes)
                r = os.path.abspath(os.path.join(
                    old_directory, c, f"{i:04}.jpg"))
                f = os.path.abspath(os.path.join(
                    new_directory, f"{number[j]:04}.jpg"))
                shutil.copy(r, f)
                l = [[f, os.path.relpath(f), c]]
                img_list += l
        csv_annotation.write_in_file(name_csv, img_list)
    except:
        logging.error(f"Failed to write")


if __name__ == "__main__":
    with open(os.path.join("Lab1", "fcc.json"), "r") as fcc_file:
        fcc = json.load(fcc_file)

    copy_with_random(fcc["main_folder"],
                     fcc["classes"], "dataset", "Lab2\csv_files\dataset_random")
