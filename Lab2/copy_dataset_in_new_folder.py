import os
import shutil
import csv
import logging
from typing import List, Tuple
from util import write_annotation_to_csv

logging.basicConfig(filename='annotation2.log', level=logging.INFO)


def get_paths(name: str) -> Tuple[List[str], List[str]]:
    """
    This function returns a tuple of absolute and relative paths for all images
    of the specific name of the animal passed to the function,
    after moving the images to another directory.
    """
    absolute_path = os.path.abspath(os.path.join('dataset2', name))
    image_paths = [os.path.join(absolute_path, img) for img in os.listdir(absolute_path)]

    relative_path = os.path.relpath(os.path.join('dataset2', name))
    relative_paths = [os.path.join(relative_path, img) for img in os.listdir(relative_path)]

    return image_paths, relative_paths


def rename_and_move_images(name: str) -> None:
    """
    This function changes the names of images by combining the image number and class
    in the format class_number.jpg,
    transfers the images to the dataset2 directory, and deletes the folder
    where the class images were stored.
    """
    relative_path = os.path.relpath('dataset2')
    class_path = os.path.join(relative_path, name)
    image_names = os.listdir(class_path)

    image_relative_paths = [os.path.join(class_path, img) for img in image_names]
    new_img_relative_paths = [os.path.join(relative_path, f'{name}_{img}') for img in image_names]

    for old_name, new_name in zip(image_relative_paths, new_img_relative_paths):
        os.replace(old_name, new_name)

    if os.path.isdir(class_path):
        shutil.rmtree(class_path)


if __name__ == "__main__":
    if os.path.isdir('dataset2'):
        shutil.rmtree('dataset2')

    old = os.path.relpath('dataset')
    new = os.path.relpath('dataset2')
    shutil.copytree(old, new)

    cat, dog = 'cat', 'dog'

    rename_and_move_images(cat)
    rename_and_move_images(dog)

    cat_absolute_paths, cat_relative_paths = get_paths(cat)
    dog_absolute_paths, dog_relative_paths = get_paths(dog)

    with open('annotation2.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\r')

        write_annotation_to_csv(writer, cat_absolute_paths, cat_relative_paths, cat)
        write_annotation_to_csv(writer, dog_absolute_paths, dog_relative_paths, dog)
