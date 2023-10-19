import os
import json
import shutil
import csv
import logging
import script1

logging.basicConfig(level=logging.INFO)


def new_csv(name_new_csv: str, classes: str, directory: str, number: int) -> None:
    try:
        row = [
            os.path.abspath(os.path.join(f"{classes}_{number:04}.jpg")),
            (os.path.join(directory, f"{classes}_{number:04}.jpg")),
            classes,
        ]
        with open(f"{name_new_csv}.csv", "a") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(row)
    except:
        logging.error(f"Failed to write data: {ex.message}\n{ex.args}\n")


def download_in_new_directory(
    old_directory: str, classes: str, new_directory: str, name_csv: str
) -> None:
    try:
        script1.make_csv(name_csv)
        for c in classes:
            count_files = len(os.listdir(os.path.join(old_directory, c)))
            for i in range(count_files):
                r = os.path.abspath(os.path.join(
                    old_directory, c, f"{i:04}.jpg"))
                f = os.path.abspath(os.path.join(
                    new_directory, f"{c}_{i:04}.jpg"))
                shutil.copy(r, f)
                new_csv(name_csv, c, new_directory, i)
    except:
        logging.error(f"Failed to write")


if __name__ == "__main__":
    with open(os.path.join("Lab1", "fcc.json"), "r") as fcc_file:
        fcc = json.load(fcc_file)

download_in_new_directory(
    fcc["main_folder"], fcc["classes"], "dataset", "dataset_new")
