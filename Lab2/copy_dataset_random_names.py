"""Module providing a function printing python version 3.11.5."""
import os
import csv
import json
import shutil
import random
import logging
from copy_dataset_in_new_folder import  create_empty_folder, create_folder
from pathlib import Path


logging.basicConfig(level=logging.DEBUG)


def get_filenames(path: str) -> None:
    """Getting a list of file names from a folder"""
    return list(str(f) for f in Path(path).rglob("*"))


def generate_random_array() -> None:
    """This func is generating a list with 10000 values (0-9999) in random order"""
    array = list(range(10000))
    random.shuffle(array)
    return array


def copy_dataset_in_new_folder(path_new_folder: str,
                               path_dataset: str,
                               annotation: str,
                               folder_for_csv: str,
                               new_name_dataset:str = 'dataset'
                               ) -> None:
    """
    Main func, that using other functions
    uploads copies of images with new names to a new folder
    """
    create_empty_folder(path_new_folder, new_name_dataset)
    try:
        create_folder(folder_for_csv)
        csv_file = open(f"{os.path.join(folder_for_csv, annotation)}.csv", 'w')
        fieldnames = ['absolute_path', 'relative_path', 'class']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        arr = generate_random_array()
        arr_iter = 0
        for f in os.scandir(path_dataset):
                if f.is_dir():
                    paths_filenames = get_filenames(f)
                    for item in paths_filenames:
                        extension = Path(item).name.split('.')[-1]
                        path_new_file = os.path.join(path_new_folder,
                                                     new_name_dataset,
                                                     f"{arr[arr_iter]}.{extension}"
                                                     )
                        shutil.copyfile(item,
                                        path_new_file
                                        )
                        writer.writerow({'absolute_path': os.path.abspath(path_new_file),
                                     'relative_path': os.path.relpath(path_new_file), 
                                     'class': f.name}
                                     )
                        arr_iter += 1
    except Exception as e:
        logging.exception(f"OS error: {e}")


if __name__ == "__main__":
    with open(os.path.join("Lab2","json","user_settings.json"), "r") as f:
        settings = json.load(f)
    copy_dataset_in_new_folder(settings["random_new_folder_for_data"],
                               settings['dataset'],
                               settings["random_copy_name_csv_file"],
                               settings["folder_for_csv_rand"]
                               )
                               
