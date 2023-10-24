"""Module providing a function printing python version 3.11.5."""
import os
import csv
import argparse
import logging
from pathlib import Path


logging.basicConfig(level=logging.DEBUG)

def get_filenames(path: str) -> None:
    """Getting a list of file names from a folder"""
    return list(str(f) for f in Path(path).rglob("*"))

def create_csv_annotation(folder: str = "dataset",
                          csv_filename: str = "annotation_dataset"
                          ) -> None:
    """Main function, that create csv annotation"""
    try:
        csv_file = open(f"{csv_filename}.scv", 'w')
        for f in os.scandir(folder):
            if f.is_dir():
                filenames = get_filenames(f)
                fieldnames = ['absolute_path', 'relative_path', 'class']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for item in filenames:
                    writer.writerow({'absolute_path': os.path.abspath(item), 'relative_path': item, 'class': f.name})
        csv_file.close()
    except Exception as e:
        logging.exception(e)
