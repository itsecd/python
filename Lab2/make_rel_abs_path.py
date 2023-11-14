import os
import csv
from typing import List
import logging

def get_full_paths(class_name: str) -> List[str]:
    """
    This function returns a list of absolute paths for all images of a certain
    the class passed to the function
    """
    try:
        full_path = os.path.abspath('dataset')
        class_path = os.path.join(full_path, class_name).replace("\\","/")
        image_names = os.listdir(class_path)
        image_full_paths = list(
            map(lambda name: os.path.join(class_path, name).replace("\\","/"), image_names))
        return image_full_paths
    except:
        logging.error(f"Failed to write")

def get_rel_paths(class_name: str) -> List[str]:
    """
    This function returns a list of relative paths relative to the dataset file for
    all images of a certain class passed to the function
    """
    try:
        rel_path = os.path.relpath('dataset')
        class_path = os.path.join(rel_path, class_name).replace("\\","/")
        image_names = os.listdir(class_path)
        image_rel_paths = list(
            map(lambda name: os.path.join(class_path, name).replace("\\","/"), image_names))
        return image_rel_paths
    except:
        logging.error(f"Failed to write")


def main() -> None:
    class1 = 'polar bear'
    class2 = 'brown bear'

    polarbear_full_paths = get_full_paths(class1)
    polarbear_rel_paths = get_rel_paths(class1)
    brownbear_full_paths = get_full_paths(class2)
    brownbear_rel_paths = get_rel_paths(class2)

    with open('paths.csv', 'w', newline='') as csv_file:  
        writer = csv.writer(csv_file, delimiter=';')
        
        # Записываем пути для первого класса
        for full_path, rel_path in zip(polarbear_full_paths, polarbear_rel_paths):
            writer.writerow([full_path, rel_path, class1])
        
        # Записываем пути для второго класса
        for full_path, rel_path in zip(brownbear_full_paths, brownbear_rel_paths):
            writer.writerow([full_path, rel_path, class2])


if __name__ == "__main__":
    main()