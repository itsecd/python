import os
import shutil
import logging
from create_annotation import get_absolute_paths
from create_annotation import get_relative_paths
from create_annotation import write_annotation_to_csv


def replace_images(class_name: str, 
                   dataset_path: str) -> None:
    """
    This function changes the names of images by combining the image number and class in the format class_number.jpg ,
    transfers the images to the dataset directory and deletes the folder where the class images were stored
    """
    try:
        abs_dataset_path = os.path.abspath(dataset_path)
        class_path = os.path.join(abs_dataset_path, class_name)
        image_names = os.listdir(class_path)
        image_full_paths = [os.path.join(class_path, name) for name in image_names]
        new_names = [f'{class_name}_{name}' for name in image_names]
        new_full_paths = [os.path.join(abs_dataset_path, new_name) for new_name in new_names]

        for old_path, new_path in zip(image_full_paths, new_full_paths):
            os.replace(old_path, new_path)

        if os.path.isdir(class_path):
            shutil.rmtree(class_path)

    except Exception as e:
        logging.error(f"Failed to write: {e}")


if __name__ == "__main__":
    class1 = 'cat'
    class2 = 'dog'
    old_path="dataset"
    dataset_path = 'dataset1'

    if os.path.isdir(dataset_path):
        shutil.rmtree(dataset_path)

    old = os.path.relpath(old_path)
    new = os.path.relpath(dataset_path)
    shutil.copytree(old, new)

    replace_images(class1, dataset_path)
    replace_images(class2, dataset_path)

    cat_absolute_paths = get_absolute_paths(class1, old_path)
    cat_relative_paths = get_relative_paths(class1, dataset_path)
    dog_absolute_paths = get_absolute_paths(class2, old_path)
    dog_relative_paths = get_relative_paths(class2, dataset_path)
    write_annotation_to_csv('annotation1.csv', cat_absolute_paths, cat_relative_paths, class1)
    write_annotation_to_csv('annotation1.csv', dog_absolute_paths, dog_relative_paths, class2)