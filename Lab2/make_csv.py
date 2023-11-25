import os
import csv
import logging
import shutil
import random
from get_way import get_path_normal, get_path_together, get_path_random


def create_csv(path_to_csv : str) -> None:
    """
    Create csv file

    Create csv file using a name_csv
    Parameters
    ----------
    path_to_csv : str
        path to csv
    """
    try:
        # if not os.path.exists(f"{name_csv}.csv"):
            with open(path_to_csv, "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(("Absolute path", "Relative path", "Class"))
    except Exception as ex:
        logging.error(f"Error create csv file: {ex}")


def write_in_csv(path_to_csv : str, img_class : str, directory : str) -> None:
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
            directory,
            os.path.relpath(directory),
            img_class
        ]
        with open(path_to_csv, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    except Exception as ex:
        logging.error(f"Error of writing row in csv: {ex}")


def make_csv(path_to_csv : str, img_classes : list, directory : str, mode : str, path_to_data : str = "") -> None:
    """
    Make new csv file

    Parameters
    ----------
    path_to_csv : str
        Save path
    img_classes : str
        List of objects
    directory : str
        Path to dataset
    """
    try:
        create_csv(path_to_csv)
        for img_class in img_classes:
            names_list = os.listdir(os.path.join(directory, img_class))
            number_of_img = len(names_list)
            for img, name in zip(range(number_of_img), names_list):
                if mode == "normal":
                    write_in_csv(path_to_csv, img_class, get_path_normal(directory, img_class, name))
                elif mode == "together":
                    shutil.copyfile(get_path_normal(directory, img_class, name), get_path_together(path_to_data, img_class, name))
                    write_in_csv(path_to_csv, img_class, get_path_together(path_to_data, img_class, name))
                elif mode == "random":
                    _random = random.randint(0, 10000)
                    shutil.copyfile(get_path_normal(directory, img_class, name), get_path_random(path_to_data, _random))
                    write_in_csv(path_to_csv, img_class, get_path_random(path_to_data, _random))
                else:
                    raise Exception("Incorrect mode")
    except Exception as ex:
        logging.error(f"Error of copy img : {img} | {ex}")