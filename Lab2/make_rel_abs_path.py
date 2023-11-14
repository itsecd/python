import os
import csv
from typing import List


def get_full_paths(class_name: str) -> List[str]:
    """
    This function returns a list of absolute paths for all images of a certain
    the class passed to the function
    """
    full_path = os.path.abspath('dataset')
    class_path = os.path.join(full_path, class_name).replace("\\","/")
    image_names = os.listdir(class_path)
    image_full_paths = list(
        map(lambda name: os.path.join(class_path, name).replace("\\","/"), image_names))
    return image_full_paths