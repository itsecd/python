import os
import shutil
import logging
import csv
from typing import List
from util import write_annotation_to_csv


logging.basicConfig(filename='annotation2.log', level=logging.INFO)


def get_paths(src_path: str, dest_path: str, name: str) -> List[str]:
    """
    This function returns a list of absolute paths for all images
    of the specific name of the animal passed to the function,
    after moving the images to another directory.
    """
    absolute_path = os.path.abspath(os.path.join(dest_path, name))
    image_paths = [os.path.join(absolute_path, img) for img in os.listdir(absolute_path)]

    return image_paths


def rename_and_move_images(src_path: str, dest_path: str, name: str) -> None:
    """
    This function changes the names of images by combining the image number and class
    in the format class_number.jpg,
    transfers the images to the destination directory, and deletes the folder
    where the class images were stored.
    """
    src_class_path = os.path.join(src_path, name)
    dest_class_path = os.path.join(dest_path, name)

    if os.path.isdir(dest_class_path):
        shutil.rmtree(dest_class_path)

    shutil.copytree(src_class_path, dest_class_path)

    image_names = os.listdir(dest_class_path)
    new_img_relative_paths = [f'{name}_{img}' for img in image_names]

    for old_name, new_name in zip(image_names, new_img_relative_paths):
        os.rename(os.path.join(dest_class_path, old_name), os.path.join(dest_class_path, new_name))

    if os.path.isdir(src_class_path):
        shutil.rmtree(src_class_path)


if __name__ == "__main__":
    src_dataset_path = 'dataset'
    dest_dataset_path = 'dataset2'

    cat, dog = 'cat', 'dog'

    rename_and_move_images(src_dataset_path, dest_dataset_path, cat)
    rename_and_move_images(src_dataset_path, dest_dataset_path, dog)

    cat_paths = get_paths(src_dataset_path, dest_dataset_path, cat)
    dog_paths = get_paths(src_dataset_path, dest_dataset_path, dog)

    with open('annotation2.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\r')

        write_annotation_to_csv(writer, cat_paths, cat_paths, cat)
        write_annotation_to_csv(writer, dog_paths, dog_paths, dog)
