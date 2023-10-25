"""Module providing a function printing python version 3.11.5."""
import os
import csv
import json
import logging
from pathlib import Path


logging.basicConfig(level=logging.DEBUG)


def get_filenames(path: str) -> None:
    """Getting a list of file names from a folder"""
    return list(str(f) for f in Path(path).rglob("*"))


def create_csv_annotation(folder: str,
                          csv_filename: str
                          ) -> None:
    """Main function, that create csv annotation"""
    try:
        csv_file = open(f"{csv_filename}.csv", 'w', newline='')
        for f in os.scandir(folder):
            if f.is_dir():
                filenames = get_filenames(f)
                fieldnames = ['absolute_path', 'relative_path', 'class']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for item in filenames:
                    writer.writerow({'absolute_path': os.path.abspath(item),
                                     'relative_path': os.path.relpath(item), 
                                     'class': f.name}
                                     )
        csv_file.close()
    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    with open(os.path.join("Lab2","user_settings.json"), "r", newline='') as f:
        settings = json.load(f)
    create_csv_annotation(settings["dataset"], settings["name_csv_file"])