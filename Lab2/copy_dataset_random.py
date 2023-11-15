import os
import shutil
import random
import logging
from make_rel_abs_path import get_full_paths
from make_rel_abs_path import get_rel_paths
from make_rel_abs_path import write_to_csv

def replace_images_and_randomize(class_name: str, 
                                 source_path: str, 
                                 dest_path: str) -> None:
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

    class1 = 'polar bear'
    class2 = 'brown bear'

    if os.path.isdir('dataset1'):
        shutil.rmtree('dataset1')

    old_path = os.path.relpath('dataset')
    new_path = os.path.relpath('dataset1')

    shutil.copytree(old_path, new_path)

    replace_images_and_randomize(class1, new_path, new_path)
    replace_images_and_randomize(class2, new_path, new_path)

    if os.path.isdir('dataset2'):
        shutil.rmtree('dataset2')

    dataset_path = 'dataset'
    polarbear_full_paths = get_full_paths(class1, dataset_path)
    polarbear_rel_paths = get_rel_paths(class1, dataset_path)
    brownbear_full_paths = get_full_paths(class2, dataset_path)
    brownbear_rel_paths = get_rel_paths(class2, dataset_path)

    write_to_csv('paths2.csv', polarbear_full_paths, polarbear_rel_paths, class1)
    write_to_csv('paths2.csv', brownbear_full_paths, brownbear_rel_paths, class2)