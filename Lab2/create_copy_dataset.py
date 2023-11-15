import os
import shutil
import csv
import json
import logging

logging.basicConfig(level=logging.INFO)


def create_annotation_file(new_copy_name: str) -> None:
    """The function takes file name and create file,  if it does not exist"""
    try:
        if not os.path.exists(new_copy_name):
            with open(f"{new_copy_name}.csv", "w", newline='') as file:
                pass
    except Exception as ex:
        logging.error(f"Failed to create file:: {ex}\n")


def create_folder(folder_name: str) -> None:
    """The function takes folder name and create folder,  if it does not exist"""
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except Exception as ex:
        logging.error(f"Failed to create folder:{ex.message}\n{ex.args}\n")


def copy_dataset_with_annotation(main_folder: str, new_copy_name: str) -> None:
    """Copy the dataset with annotations to a new directory and create an annotation file.
        main_folder (str): The path to the main folder where the dataset is located.
        new_copy_name (str): The name of the new directory to which the dataset will be copied.
    """
    create_folder(new_copy_name)

    annotations = []

    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".jpg"):
                query = os.path.basename(root)
                destination_folder = os.path.join(new_copy_name, query)
                create_folder(destination_folder)

                new_filename = f"{query}_{len(os.listdir(destination_folder)) + 0:04d}.jpg"
                source_filepath = os.path.join(root, file)
                destination_filepath = os.path.join(
                    destination_folder, new_filename)
                shutil.copyfile(source_filepath, destination_filepath)
                annotations.append(
                    [source_filepath, destination_filepath, query])

    try:
        create_annotation_file(new_copy_name)
        with open(f"{new_copy_name}.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(annotations)
    except Exception as ex:
        logging.error(f"Failed to write: {ex}\n")


if __name__ == "__main__":
    with open("Lab2/options.json", "r") as options_file:
        options = json.load(options_file)
        copy_dataset_with_annotation(
            options["main_folder"], options["new_copy_name"])
