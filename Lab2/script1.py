import csv
import os
import json
import logging

logging.basicConfig(level=logging.INFO)


def make_csv(name_csv: str) -> None:
    '''Function creates a csv file
    with the specified name'''
    try:
        if not os.path.exists(name_csv):
            with open(f"{name_csv}.csv", "w") as file:
                writer = csv.writer(file)
                writer.writerow(("Absolute path", "Relative path", "Class"))
    except Exception as ex:
        logging.error(f"Couldn't create file: {ex.message}\n{ex.args}\n")


def write_in_file(name_csv: str, classes: str, directory: str) -> None:
    '''Accepts dataset's classes,
    counts the number of files in each class folder,
    gets data for each image and writes them to csv'''
    try:
        make_csv(name_csv)
        for c in classes:
            count_files = len(os.listdir(os.path.join(directory, c)))
            for img in range(count_files):
                rows = [os.path.abspath(os.path.join(
                    c, f"{img:04}.jpg")), (os.path.join(directory,
                                                        c, f"{img:04}.jpg")), c]
                with open(f"{name_csv}.csv", "a") as file:
                    writer = csv.writer(file, lineterminator="\n")
                    writer.writerow(
                        rows
                    )
    except:
        logging.error(f"Failed to write data: {ex.message}\n{ex.args}\n")


if __name__ == "__main__":

    with open(os.path.join("Lab1", "fcc.json"), "r") as fcc_file:
        fcc = json.load(fcc_file)

    write_in_file("datasets", fcc["classes"], fcc["main_folder"])
