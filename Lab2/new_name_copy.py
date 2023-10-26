import os
import json
import shutil
import logging
import csv_annotation


logging.basicConfig(level=logging.INFO)


def copy_in_new_directory(
    old_directory: str, classes: str, new_directory: str, name_csv: str
) -> None:
    """The function copies images from class folders to the dataset.
    New file name=class+number"""
    try:
        img_list = list()
        for c in classes:
            count_files = len(os.listdir(os.path.join(old_directory, c)))
            for i in range(count_files):
                r = os.path.abspath(os.path.join(
                    old_directory, c, f"{i:04}.jpg"))
                f = os.path.abspath(os.path.join(
                    new_directory, f"{c}_{i:04}.jpg"))
                shutil.copy(r, f)
                l = [[f, os.path.relpath(f), c]]
                img_list += l
        csv_annotation.write_in_file(name_csv, img_list)
    except:
        logging.error(f"Failed to write")


if __name__ == "__main__":
    with open(os.path.join("Lab1", "fcc.json"), "r") as fcc_file:
        fcc = json.load(fcc_file)

    copy_in_new_directory(
        fcc["main_folder"], fcc["classes"], "dataset", "Lab2\csv_files\dataset_new"
    )
