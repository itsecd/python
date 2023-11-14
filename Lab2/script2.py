import os
import shutil
import csv
from typing import List

def get_absolute_path2(name: str) -> List[str]:
    """
    This function returns a list of absolute paths for all images of a specific animal name,
    passed to the function after moving the images to another directory
    """
    absolute_path = os.path.abspath('dataset2')
    image_names = os.listdir(absolute_path)

    image_class_names = [name_img for name_img in image_names if name in name_img]

    image_absolute_paths = list(
        map(lambda name: os.path.join(absolute_path, name), image_class_names))
    return image_absolute_paths


def get_relative_path2(name: str) -> List[str]:
    """
    This function returns a list of relative paths for all images of a specific animal name
    passed to the function after moving the images to another directory
    """
    relative_path = os.path.relpath('dataset2')
    image_names = os.listdir(relative_path)

    image_class_names = [name_img for name_img in image_names if name in name_img]

    image_relative_paths = list(
        map(lambda name: os.path.join(relative_path, name), image_class_names))
    return image_relative_paths


def replace_images(name: str) -> None:
    """
    This function changes the names of images by combining the image number and class in the format class_number.jpg ,
    transfers the images to the dataset2 directory and deletes the folder where the class images were stored
    """
    relative_path = os.path.relpath('dataset2')
    class_path = os.path.join(relative_path, name)
    image_names = os.listdir(class_path)

    image_relative_paths = list(map(lambda img: os.path.join(class_path, img), image_names))
    new_img_relative_paths = list(map(lambda img: os.path.join(relative_path, f'{name}_{img}'), image_names))

    for old_name, new_name in zip(image_relative_paths, new_img_relative_paths):
        os.replace(old_name, new_name)

    os.chdir('dataset2')
    if os.path.isdir(name):
        os.rmdir(name)
    os.chdir('..')


def main() -> None:
    cat='cat'
    dog='dog'
    if os.path.isdir('dataset2'):
        shutil.rmtree('dataset2')
    old = os.path.relpath('dataset')
    new = os.path.relpath('dataset2')
    shutil.copytree(old, new)

    replace_images(cat)
    replace_images(dog) 
    
    cat_absolute_paths = get_absolute_path2(cat)
    cat_relative_paths = get_relative_path2(cat)
    dog_absolute_paths = get_absolute_path2(dog)
    dog_relative_paths = get_relative_path2(dog)

    with open('annotation2.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\r')

        for absolute_path, relative_path in zip(cat_absolute_paths, cat_relative_paths):
            writer.writerow([absolute_path, relative_path, cat])

        for absolute_path, relative_path in zip(dog_absolute_paths, dog_relative_paths):
            writer.writerow([absolute_path, relative_path, dog])


if __name__ == "__main__":
    main()