import os
import shutil
import csv
import random
import logging


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


def write_to_csv(
        file_path: str, 
        full_paths: list, 
        rel_paths: list, 
        dataset_path: str) -> None:
    """
    This function writes the paths to a CSV file.
    """
    with open(file_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=';', lineterminator='\r')
        for full_path, rel_path in zip(full_paths, rel_paths):
            class_name = rel_path.split("_")[0].replace(os.path.join(dataset_path, ""), "")
            writer.writerow([full_path, rel_path, class_name])    



if __name__ == "__main__":
    if os.path.isdir('dataset1'):
        shutil.rmtree('dataset1')

    old_path = os.path.relpath('dataset')
    new_path = os.path.relpath('dataset1')

    shutil.copytree(old_path, new_path)

    replace_images_and_randomize('brown bear', new_path, new_path)
    replace_images_and_randomize('polar bear', new_path, new_path)

    if os.path.isdir('dataset2'):
        shutil.rmtree('dataset2')

    old_path = os.path.relpath('dataset1')
    new_path = os.path.relpath('dataset2')

    shutil.copytree(old_path, new_path)

    new_names = os.listdir(new_path)
    new_rel_paths = [os.path.join(new_path, name) for name in new_names]
    new_full_paths = [os.path.join(os.path.abspath(new_path), name) for name in new_names]

    write_to_csv('paths2.csv', new_full_paths, new_rel_paths, new_path)