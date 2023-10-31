import os
import csv
import logging
import shutil
import random
from get_way import get_path_normal, get_path_together, get_path_random


def create_csv(name_csv : str) -> None:
    """
    Create csv file

    Create csv file using a name_csv
    Parameters
    ----------
    name_csv : str
        Name of csv file
    """
    try:
        # if not os.path.exists(f"{name_csv}.csv"):
            with open(f"{name_csv}.csv", "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(("Absolute path", "Relative path", "Class"))
    except Exception as ex:
        logging.error(f"Error create csv file: {ex}")


def write_in_csv(name_csv : str, img_class : str, directory : str) -> None:
    """
    Write in csv file

    Parameters
    ----------
    name_csv : str
        Name of csv file
    img_class : str
        Name of object
    directory : str
        Directory where is our img
    """
    try:    
        row = [
            os.path.abspath(directory),
            directory,
            img_class
        ]
        with open(f"{name_csv}.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    except Exception as ex:
        logging.error(f"Error of writing row in csv: {ex}")


def make_csv(name_csv : str, img_classes : list, directory : str, mode : str) -> None:
    """
    Make new csv file

    Parameters
    ----------
    name_csv : str
        Name of csv file
    img_classes : str
        List of objects
    directory : str
        Path to dataset
    """
    try:
        create_csv(name_csv)
        for img_class in img_classes:
            number_of_img = len(os.listdir(os.path.join(directory, img_class)))
            for img in range(number_of_img):
                if mode == "normal":
                    write_in_csv(name_csv, img_class, get_path_normal(img_class, img))
                elif mode == "together":
                    shutil.copyfile(get_path_normal(img_class, img), get_path_together(img_class, img))
                    write_in_csv(name_csv, img_class, get_path_together(img_class, img))
                elif mode == "random":
                    _random = random.randint(0, 10000)
                    shutil.copyfile(get_path_normal(img_class, img), get_path_random(img_class, _random))
                    write_in_csv(name_csv, img_class, get_path_random(img_class, _random))
                else:
                    raise Exception("Incorrect mode")
    except Exception as ex:
        logging.error(f"Error of copy img : {img} | {ex}")