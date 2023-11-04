"""Module providing a function printing python version 3.11.5."""
import os
import csv
import json
import logging
from copy_dataset_in_new_folder import  create_folder
from pathlib import Path


logging.basicConfig(level=logging.DEBUG)


def get_filenames(path: str) -> None:
    """Getting a list of file names from a folder"""
    return list(str(f) for f in Path(path).rglob("*"))


def create_csv_annotation(folder: str,
                          csv_filename: str,
                          folder_for_csv: str
                          ) -> None:
    """Main function, that create csv annotation"""
    try:
        create_folder(folder_for_csv)
        with open(f"{os.path.join(folder_for_csv, csv_filename)}.csv", 'w', newline='') as csv_file:
            fieldnames = ['absolute_path', 'relative_path', 'class']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for f in os.scandir(folder):
                if f.is_dir():
                    filenames = get_filenames(f)
                    for item in filenames:
                        writer.writerow({'absolute_path': os.path.abspath(item),
                                        'relative_path': os.path.relpath(item), 
                                        'class': f.name}
                                        )
    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    with open(os.path.join("Lab2","json","user_settings.json"), "r", newline='') as f:
        settings = json.load(f)
    create_csv_annotation(settings["dataset"],
                          settings["name_csv_file"],
                          settings["folder_for_csv"])
