import csv
import os
import json
import logging


logging.basicConfig(level=logging.INFO)


def make_csv(name_csv: str) -> None:
    """Function creates a csv file
    with the specified name"""
    try:
        if not os.path.exists(name_csv):
            with open(f"{name_csv}.csv", "a") as file:
                csv.writer(file, lineterminator="\n")
    except Exception as ex:
        logging.error(f"Couldn't create file: {ex.message}\n{ex.args}\n")


def make_list(directory: str, classes: str) -> list:
    """Creates a list with parameters for a csv file"""
    img_list = list()
    for c in classes:
        count_files = len(os.listdir(os.path.join(directory, c)))
        for i in range(count_files):
            r = [
                [os.path.abspath(os.path.join(directory, c, f"{i:04}.jpg")),
                    os.path.join(directory, c, f"{i:04}.jpg"),
                    c,]
            ]
            img_list += r
    return img_list


def write_in_file(name_csv: str, img_list: list) -> None:
    """Accepts dataset's classes,
    gets data for each image and writes them to csv"""
    try:
        make_csv(name_csv)
        for img in img_list:
            with open(f"{name_csv}.csv", "a") as file:
                writer = csv.writer(file, lineterminator="\n")
                writer.writerow(img)
    except:
        logging.error(f"Failed to write data: {ex.message}\n{ex.args}\n")


if __name__ == "__main__":
    with open(os.path.join("Lab1", "fcc.json"), "r") as fcc_file:
        fcc = json.load(fcc_file)

    l = make_list("dataset", fcc["classes"])
    write_in_file("Lab2\csv_files\datasets", l)
