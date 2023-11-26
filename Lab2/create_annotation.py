import os
import csv
from typing import List
import logging


logging.basicConfig(filename='annotation.log', level=logging.INFO)

import os
from typing import List

def get_absolute_paths(class_name: str, dataset_path: str) -> List[str]:
    """
    This function returns a list of absolute paths for all images of a certain
    the class passed to the function
    """
    try:
        absolute_path = os.path.abspath(dataset_path)
        class_path = os.path.join(absolute_path, class_name).replace("\\", "/")
        
        if os.path.exists(class_path):
            image_names = os.listdir(class_path)
            image_absolute_paths = [os.path.join(class_path, name).replace("\\", "/") for name in image_names]
            return image_absolute_paths
        else:
            image_names = os.listdir(absolute_path)
            image_class_names = [name for name in image_names if class_name in name]
            image_absolute_paths = [os.path.join(absolute_path, name) for name in image_class_names]
            return image_absolute_paths

    except Exception as e:
        logging.error(f"Failed to get_absolute_paths: {e}")



def get_relative_paths(class_name: str, 
                  dataset_path: str) -> List[str]:
    """
    This function returns a list of relative paths relative to the dataset file for
    all images of a certain class passed to the function
    """
    try:
        relative_path = os.path.relpath(dataset_path)
        class_path = os.path.join(relative_path, class_name).replace("\\", "/")
        if os.path.exists(class_path):
            image_names = os.listdir(class_path)
            image_relative_paths = list(
                map(lambda name: os.path.join(class_path, name).replace("\\", "/"), image_names))
            return image_relative_paths
        else:
            image_names = os.listdir(relative_path)
            image_class_names = [name for name in image_names if class_name in name]
            image_relative_paths = list(
                map(lambda name: os.path.join(relative_path, name), image_class_names))
            return image_relative_paths
    except Exception as e:
        logging.error(f"Failed to write_relative: {e}")

def write_annotation_to_csv(file_path: str, 
                            absolute_paths: List[str], 
                            relative_paths: List[str], 
                            class_name: str) -> None:
    try:
        with open(file_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';', lineterminator='\r')
            for absolute_path, relative_path in zip(absolute_paths, relative_paths):
                writer.writerow([absolute_path, relative_path, class_name])
    except Exception as e:
        logging.error(f"Failed to write_annotation_to_csv: {e}")
        

if __name__ == "__main__":
    cat, dog = 'cat', 'dog'

    cat_absolute_paths = get_absolute_paths(cat)
    cat_relative_paths = get_relative_paths(cat)
    dog_absolute_paths = get_absolute_paths(dog)
    dog_relative_paths = get_relative_paths(dog)

    annotation_file = 'annotation.csv'

    for absolute_paths, relative_paths, class_name in zip([cat_absolute_paths, dog_absolute_paths],
                                                 [cat_relative_paths, dog_relative_paths],
                                                 [cat, dog]):
        write_annotation_to_csv(annotation_file, absolute_paths, relative_paths, class_name)
