import os
import csv 
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

def create_folder(path_new_folder: str, base_folder: str) -> None:
    path = os.path.join(path_new_folder, base_folder)
    try:
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)
    except OSError as e:
        logging.exception(f"OS error: {e}")

def get_filenames(path: str) -> None:
    """Getting a list of file names from a folder"""
    return list(str(f) for f in Path(path).rglob("*"))

def copy_dataset_in_new_folder(path_new_folder: str, path_dataset: str = 'dataset') -> None:
    create_folder(path_new_folder, path_dataset)
    try:
        csv_file = open("new_data_annotation.scv", 'w')
        for f in os.scandir(path_dataset):
                if f.is_dir():
                    paths_filenames = get_filenames(f)
                    fieldnames = ['absolute_path', 'relative_path', 'class']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in paths_filenames:
                        filename = Path(item).name
                        path_new_file = os.path.join(path_new_folder,
                                                     'dataset',
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
