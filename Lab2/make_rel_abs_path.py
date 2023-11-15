import os
import csv
from typing import List
import logging


def get_full_paths(class_name: str, 
                   dataset_path: str) -> List[str]:
    """
    This function returns a list of absolute paths for all images of a certain
    the class passed to the function
    """
    try:
        full_path = os.path.abspath(dataset_path)
        class_path = os.path.join(full_path, class_name).replace("\\", "/")
        image_names = os.listdir(class_path)
        image_full_paths = list(
            map(lambda name: os.path.join(class_path, name).replace("\\", "/"), image_names))
        return image_full_paths
    except Exception as e:
        logging.error(f"Failed to write_abs: {e}")


def get_rel_paths(class_name: str, 
                  dataset_path: str) -> List[str]:
    """
    This function returns a list of relative paths relative to the dataset file for
    all images of a certain class passed to the function
    """
    try:
        rel_path = os.path.relpath(dataset_path)
        class_path = os.path.join(rel_path, class_name).replace("\\", "/")
        image_names = os.listdir(class_path)
        image_rel_paths = list(
            map(lambda name: os.path.join(class_path, name).replace("\\", "/"), image_names))
        return image_rel_paths
    except Exception as e:
        logging.error(f"Failed to write_rel: {e}")


def write_to_csv(file_path: str, 
                 full_paths: List[str], 
                 rel_paths: List[str], 
                 class_name: str) -> None:
    try:
        with open(file_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';', lineterminator='\r')
            for full_path, rel_path in zip(full_paths, rel_paths):
                writer.writerow([full_path, rel_path, class_name])
    except Exception as e:
        logging.error(f"Failed to write_csv: {e}")


if __name__ == "__main__":
    class1 = 'polar bear'
    class2 = 'brown bear'

    dataset_path = 'dataset'
    polarbear_full_paths = get_full_paths(class1, dataset_path)
    polarbear_rel_paths = get_rel_paths(class1, dataset_path)
    brownbear_full_paths = get_full_paths(class2, dataset_path)
    brownbear_rel_paths = get_rel_paths(class2, dataset_path)

    write_to_csv('paths.csv', polarbear_full_paths, polarbear_rel_paths, class1)
    write_to_csv('paths.csv', brownbear_full_paths, brownbear_rel_paths, class2)
