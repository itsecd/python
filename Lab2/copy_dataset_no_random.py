import os
import shutil
import csv


def get_full_paths1(class_name: str) -> list:
    """
    This function returns a list of absolute paths for all images of a certain
    the class passed to the function after moving the images to another directory
    """
    full_path = os.path.abspath('dataset1')
    image_names = os.listdir(full_path)
    image_class_names = [name for name in image_names if class_name in name]
    image_full_paths = list(
        map(lambda name: os.path.join(full_path, name), image_class_names))
    return image_full_paths


def get_rel_paths1(class_name: str) -> list:
    """
    This function returns a list of relative paths for all images of a certain class
    passed to the function after moving the images to another directory
    """
    rel_path = os.path.relpath('dataset1')
    image_names = os.listdir(rel_path)
    image_class_names = [name for name in image_names if class_name in name]
    image_rel_paths = list(
        map(lambda name: os.path.join(rel_path, name), image_class_names))
    return image_rel_paths