import os
import shutil
import csv
import random
import logging
from create_annotation import get_absolute_paths
from create_annotation import get_relative_paths
from create_annotation import write_annotation_to_csv


logging.basicConfig(filename='annotation3.log', level=logging.INFO)


def process_images(class_name: str, source_path: str, dest_path: str) -> None:
    """
    This function renames images by combining the image number, class name, and random number
    in the format class_name_number.jpg, transfers the images to the destination directory,
    and deletes the folder where the class images were stored.
    """
    try:
        abs_source_path = os.path.abspath(source_path)
        class_path = os.path.join(abs_source_path, class_name)
        image_names = os.listdir(class_path)
        image_full_paths = [os.path.join(class_path, name) for name in image_names]

        random_numbers = random.sample(range(0, 10001), len(image_names))
        new_names = [f'{class_name}_{random_number}.jpg' for random_number in random_numbers]
        new_full_paths = [os.path.join(dest_path, name) for name in new_names]

        for old_path, new_path in zip(image_full_paths, new_full_paths):
            os.replace(old_path, new_path)

        if os.path.isdir(class_path):
            shutil.rmtree(class_path)

    except Exception as e:
        logging.error(f"Failed to write: {e}")


if __name__ == "__main__":

    class1 = 'cat'
    class2 = 'dog'

    if os.path.isdir('dataset1'):
        shutil.rmtree('dataset1')

    old_path = os.path.relpath('dataset')
    new_path = os.path.relpath('dataset1')

    shutil.copytree(old_path, new_path)

    process_images(class1, new_path, new_path)
    process_images(class2, new_path, new_path)

    if os.path.isdir('dataset2'):
        shutil.rmtree('dataset2')

    dataset_path = 'dataset'
    cat_absolute_paths = get_absolute_paths(class1, dataset_path)
    cat_relative_paths = get_relative_paths(class1, dataset_path)
    dog_absolute_paths = get_absolute_paths(class2, dataset_path)
    dog_relative_paths = get_relative_paths(class2, dataset_path)

    write_annotation_to_csv('annotation2.csv', cat_absolute_paths, cat_relative_paths, class1)
    write_annotation_to_csv('annotation2.csv', dog_absolute_paths, dog_relative_paths, class2)