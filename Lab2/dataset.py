import argparse
import logging
import os
import uuid
import shutil


def copy_dataset(src_folder: str, dest_folder: str, randomize: bool = False) -> None:
    """
    Copies and renames or randomizes dataset.

    Parameters
    ----------
    src_folder (str): The path to the source folder containing the dataset.
    dest_folder (str): The path to the destination folder where the dataset will be copied.
    randomize (bool, optional): If True, the dataset will be copied with randomized filenames;
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
                        unique_id = str(uuid.uuid4())[:8]
                        dest_filename = f"{unique_id}.jpg"
                    else:
                        dest_filename = f"{class_folder}_{idx:04}.jpg"

                    dest_filepath = os.path.join(dest_folder, dest_filename)
                    shutil.copy(src_filepath, dest_filepath)

        logging.info(f"Dataset copied and {'randomized' if randomize else 'uniquely named'}")
    except (FileNotFoundError, PermissionError) as e:
        logging.exception(f"Error in copying the dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy and rename or randomize dataset.')
    parser.add_argument('src_folder', type=str, help='Source folder path')
    parser.add_argument('dest_folder', type=str, help='Destination folder path')
    parser.add_argument('--randomize', action='store_true', help='Assign random numbers?')

    args = parser.parse_args()

    copy_dataset(args.src_folder, args.dest_folder, randomize=args.randomize)