import os
import shutil
import csv
import json
import logging
import random

logging.basicConfig(level=logging.INFO)

def create_annotation_file(annotations: list, file_path: str) -> None:
    """Create an annotation file.
    annotations (list): List of annotations to be written to the file.
    file_path (str): The path to the annotation file where the annotations will be written.
    """
    try:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(annotations)
    except Exception as ex:
        logging.error(f"Failed to write the annotation file: {ex}\n")

def copy_dataset_with_annotation(main_folder: str, random_copy_name: str) -> None:
    """Copy the dataset with annotations to a new directory and create an annotation file.
    main_folder (str): The path to the main folder where the dataset is located.
    random_copy_name (str): The name of the new directory to which the dataset will be copied.
    """
    try:
        os.makedirs(random_copy_name, exist_ok=True)

        annotations = []

        for root, dirs, files in os.walk(main_folder):
            for file in files:
                if file.endswith(".jpg"):
                    random_number = random.randint(0, 10000)
                    new_filename = f"{random_number}.jpg"
                    source_filepath = os.path.join(root, file)
                    query = os.path.basename(root)
                    destination_folder = os.path.join(random_copy_name, query)
                    os.makedirs(destination_folder, exist_ok=True)
                    destination_filepath = os.path.join(destination_folder, new_filename)
                    shutil.copyfile(source_filepath, destination_filepath)
                    annotations.append([source_filepath, destination_filepath, query])

        annotation_file_path = os.path.join(random_copy_name, "annotations.csv")
        create_annotation_file(annotations, annotation_file_path)

    except Exception as ex:
        logging.error(f"Error while copying the dataset: {ex}\n")

if __name__ == "__main__":
    with open("Lab2/options.json", "r") as options_file:
        options = json.load(options_file)
        copy_dataset_with_annotation(options["main_folder"], options["random_copy_name"])