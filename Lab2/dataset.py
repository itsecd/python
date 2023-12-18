import argparse
import logging
import os
import random
import shutil


def copy_dataset(src_folder: str, dest_folder: str, randomize: bool = False) -> None:
    """
    Copies and renames or randomizes dataset.

    Parameters
    ----------
    src_folder : str
    dest_folder : str
    randomize : bool, optional
    """
    try:
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        for class_folder in os.listdir(src_folder):
            class_path = os.path.join(src_folder, class_folder)
            if os.path.isdir(class_path):
                for idx, filename in enumerate(os.listdir(class_path)):
                    src_filepath = os.path.join(class_path, filename)

                    if randomize:
                        dest_filename = f"{random.randint(0, 10000)}.jpg"
                    else:
                        dest_filename = f"{class_folder}_{idx:04}.jpg"

                    dest_filepath = os.path.join(dest_folder, dest_filename)

                    shutil.copy(src_filepath, dest_filepath)

        logging.info(f"Dataset copied and {'randomized' if randomize else 'renamed'}")
    except Exception as e:
        logging.error(f"Error {'randomizing' if randomize else 'renaming'} dataset: {e}")