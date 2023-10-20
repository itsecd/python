import csv
import json
import os
import logging
from get_way import get_path_normal

SRC = "Lab2/dataset"

def create_csv(name_csv : str) -> None:
    try:
        if not os.path.exists(name_csv):
            with open(f"{name_csv}.csv", "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(("Absolute path", "Relative path", "Class"))
    except Exception as ex:
        logging.error(f"Error create csv file: {ex}")

def write_in_file(name_csv : str, img_classes : list, directory : str) -> None:
    try:
        create_csv(name_csv)
        for img_class in img_classes:
            number_of_img = len(os.listdir(os.path.join(directory, img_class)))
            for img in range(number_of_img):
                write_in_csv(name_csv, img_class, get_path_normal(img_class, img))
    except Exception as ex:
        logging.error(f"Error of add to csv file : {img} | {ex}")

def write_in_csv(name_csv : str, img_class : str, directory : str) -> None:
    try:    
        row = [
            os.path.abspath(directory).replace("\\", "/"),
            directory.replace("\\", "/"),
            img_class
        ]
        with open(f"{name_csv}.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    except Exception as ex:
        logging.error(f"Error of writing row in csv: {ex}")


if __name__ == '__main__':
    with open(os.path.join("Lab1", "input_data.json"), 'r') as fjson:
        fj = json.load(fjson)

    with open(os.path.join("Lab2", "src_csv.json"), 'r') as srcjson:
        sj = json.load(srcjson)

    logging.basicConfig(level=logging.INFO)

    write_in_file(os.path.join("Lab2", sj["normal"]), fj["objects"], fj["main_folder"])