import csv
import json
import os
import logging

def create_csv(name_csv : str) -> None:
    try:
        if not os.path.exists(name_csv):
            with open(f"{name_csv}.csv", "w") as file:
                writer = csv.writer(file)
                writer.writerow(("Absolute path", "Relative path", "Class"))
    except Exception as ex:
        logging.error(f"Error create csv file: {ex}")

def write_in_file(name_csv : str, img_class : str, directory : str) -> None:
    try:
        create_csv(name_csv)
        for img in range(len(os.listdir(os.path.join(directory, img_class)))):
            row = [
                os.path.abspath(os.path.join(directory, img_class, f"{img:04}.jpg")),
                os.path.join(directory, img_class, f"{img:04}.jpg"),
                img_class
            ]
            with open(f"{name_csv}.csv", "a") as file:
                writer = csv.writer(file)
                writer.writerow(row)
    except Exception as ex:
        logging.error(f"Error of add to csv file : {img} | {ex}")


if __name__ == '__main__':
    with open(os.path.join("Lab1", "input_data.json"), 'r') as fjson:
        fj = json.load(fjson)

    logging.basicConfig(level=logging.INFO)
    write_in_file("Lab2/dataset", fj["object"], fj["main_folder"])