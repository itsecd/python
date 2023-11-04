"""Module providing a function printing python version 3.11.5."""
import os
import csv
import json
import shutil
import logging
from pathlib import Path


logging.basicConfig(level=logging.DEBUG)


def create_empty_folder(path_new_folder: str, base_folder: str) -> None:
    """This func create folder, to which we upload copies"""
    path = os.path.join(path_new_folder, base_folder)
    try:
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)
    except OSError as e:
        logging.exception(f"OS error: {e}")


def create_folder(path: str):
    """This func create folder"""
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError:
        logging.exception(f"OS error: {OSError}")        


def get_filenames(path: str) -> None:
    """Getting a list of file names from a folder"""
    return list(str(f) for f in Path(path).rglob("*"))


def copy_dataset_in_new_folder(path_new_folder: str,
                               path_dataset: str,
                               annotation: str,
                               folder_for_csv: str) -> None:
    """Main func, that using other functions uploads copies of images with new names to a new folder"""
    create_empty_folder(path_new_folder, path_dataset)
    try:
        create_folder(folder_for_csv)
        csv_file = open(f"{os.path.join(folder_for_csv, annotation)}.csv", 'w', newline='')
        fieldnames = ['absolute_path', 'relative_path', 'class']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for f in os.scandir(path_dataset):
                if f.is_dir():
                    paths_filenames = get_filenames(f)
                    for item in paths_filenames:
                        filename = Path(item).name
                        path_new_file = os.path.join(path_new_folder,
                                                     path_dataset,
                                                     f"{f.name}_{filename}"
                                                     )
                        shutil.copyfile(item,
                                        path_new_file
                                        )
                        writer.writerow({'absolute_path': os.path.abspath(path_new_file),
                                     'relative_path': os.path.relpath(path_new_file), 
                                     'class': f.name}
                                     )
    except Exception as e:
        logging.exception(f"OS error: {e}")


if __name__ == "__main__":
    with open(os.path.join("Lab2","json","user_settings.json"), "r", newline='') as f:
        settings = json.load(f)
    copy_dataset_in_new_folder(settings["new_folder_for_data"],
                               settings['dataset'],
                               settings["copy_name_csv_file"],
                               settings["folder_for_csv_copy"])
