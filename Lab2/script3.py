import random
import os
import json
import shutil
import csv
import logging
import script1


logging.basicConfig(level=logging.INFO)


def new_csv(name_csv: str, classes: str, directory: str, number: int) -> None:
    '''the function creates a new csv file
    that records the updated dataset'''
    try:
        row = [
            os.path.abspath(os.path.join(f"{number:04}.jpg")),
            (os.path.join(directory, f"{number:04}.jpg")),
            classes,
        ]
        with open(f"{name_csv}.csv", "a") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(row)
    except:
        logging.error(f"Failed to write data: {ex.message}\n{ex.args}\n")


def download_with_random(
    old_directory: str, classes: str, new_directory: str, name_csv: str
) -> None:
    '''The function copies images from class folders to the dataset.
    New file name=random number'''
    try:
        script1.make_csv(name_csv)
        numbers = set()
        count_files = len(os.listdir(os.path.join(old_directory, classes[0])))
        while len(numbers) <= (count_files * len(classes)):
            numbers.add(random.randint(0, 10000))
        number = list(numbers)
        for c in classes:
            for i in range(count_files):
                # по условию папки с исходными изображениями хранятся в dataset,
                # поэтому при подсчете количества изображений необходимо вычитать количество этих папок из общего количества файлов в dataset
                j = len(os.listdir(os.path.join(new_directory))) - len(classes)
                r = os.path.abspath(os.path.join(
                    old_directory, c, f"{i:04}.jpg"))
                f = os.path.abspath(os.path.join(
                    new_directory, f"{number[j]:04}.jpg"))
                shutil.copy(r, f)
                new_csv(name_csv, c, new_directory, number[j])

    except:
        logging.error(f"Failed to write")


if __name__ == "__main__":
    with open(os.path.join("Lab1", "fcc.json"), "r") as fcc_file:
        fcc = json.load(fcc_file)

    download_with_random(fcc["main_folder"],
                         fcc["classes"], "dataset", "random")
