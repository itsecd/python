import os
import csv
import json
import logging

logging.basicConfig(level=logging.INFO)


def list_files_in_directory(main_folder: str, allowed_extensions: list = [".jpg", ".jpeg", ".png"]) -> list:
    """Get a list of paths to all files with specified extensions in the specified folder and its subfolders.
    main_folder (str): The path to the main folder where files need to be found.
    allowed_extensions (list): List of allowed file extensions. Default is ['.jpg', '.jpeg', '.png'].
    list: A list of paths to all files with the specified extensions in the specified folder and its subfolders.
    """
    file_list = []
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in allowed_extensions):
                file_list.append(os.path.join(root, file))
    return file_list


def generate_annotation(main_folder: str) -> list:
    """Generate annotations for all files with the .jpg extension in the specified folder and its subfolders.
    main_folder (str): The path to the main folder where the files are located.
    list(str): A list of annotations for each file with the absolute path, relative path, and query label.
    """
    file_list = list_files_in_directory(
        main_folder, allowed_extensions=[".jpg", ".png", ".gif"])
    annotations = []
    for file in file_list:
        absolute_path_from_project = os.path.abspath(file)
        relative_path_from_project = os.path.relpath(file, main_folder)
        query_label = os.path.basename(os.path.dirname(file))
        annotations.append([absolute_path_from_project,
                           relative_path_from_project, query_label])
    return annotations


def create_annotation_file(annotation_file: str) -> None:
    """The function takes file name and create file,  if it does not exist"""
    try:
        if not os.path.exists(annotation_file):
            with open(f"{annotation_file}.csv", "w", newline=''):
                pass
    except Exception as ex:
        logging.error(f"Failed to create file:: {ex}\n")


def write_annotation_to_csv(main_folder: str, annotation_file: str) -> None:
    """Write annotations to a CSV file for all files with the .jpg extension in the specified folder and its subfolders.
    main_folder (str): The path to the main folder where the files are located.
    annotation_file (str): The name of the annotation file to be created.
    """
    try:
        create_annotation_file(annotation_file)
        annotations = generate_annotation(main_folder)
        with open(f"{annotation_file}.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(annotations)
    except:
        logging.error("Failed to write data: {ex}\n")


if __name__ == "__main__":
    with open("Lab2/options.json", "r") as options_file:
        options = json.load(options_file)
        write_annotation_to_csv(
            options["main_folder"], options["annotation_file"])
