import os
import shutil
import csv
import logging

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


def replace_images(class_name: str) -> list:
    """
    This function changes the names of images by combining the image number and class in the format class_number.jpg ,
    transfers the images to the dataset directory and deletes the folder where the class images were stored
    """
    try:
        rel_path = os.path.relpath('dataset1')
        class_path = os.path.join(rel_path, class_name)
        image_names = os.listdir(class_path)
        image_rel_paths = list(
            map(lambda name: os.path.join(class_path, name), image_names))
        new_rel_paths = list(
            map(lambda name: os.path.join(rel_path, f'{class_name}_{name}'), image_names))
        for old_name, new_name in zip(image_rel_paths, new_rel_paths):
            os.replace(old_name, new_name)

        os.chdir('dataset1')

        if os.path.isdir(class_name):
            os.rmdir(class_name)

        os.chdir('..')
    except:
        logging.error(f"Failed to write")


def main() -> None:

    class1 = 'polar bear'
    class2 = 'brown bear'

    if os.path.isdir('dataset1'):
        shutil.rmtree('dataset1')

    old = os.path.relpath('dataset')
    new = os.path.relpath('dataset1')
    shutil.copytree(old, new)

    replace_images(class1)
    replace_images(class2)

    polarbear_full_paths = get_full_paths1(class1)
    polarbear_rel_paths = get_rel_paths1(class1)
    brownbear_full_paths = get_full_paths1(class2)
    brownbear_rel_paths = get_rel_paths1(class2)
    
    with open('paths1.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=';', lineterminator='\r')
        for full_path, rel_path in zip(polarbear_full_paths, polarbear_rel_paths):
            writer.writerow([full_path, rel_path, class1])
        for full_path, rel_path in zip(brownbear_full_paths, brownbear_rel_paths):
            writer.writerow([full_path, rel_path, class2])


if __name__ == "__main__":
    main()