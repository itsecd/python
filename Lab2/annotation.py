import argparse
import csv
import logging
import os
from typing import List, Tuple

ABSOLUTE_PATH_COLUMN = 'The absolute path'
RELATIVE_PATH_COLUMN = 'Relative path'
CLASS_NAME_COLUMN = 'The text name of the class'


def create_annotation_file(folder_path: str, subfolder_paths: List[str], annotation_file_path: str) -> Tuple[str, str, str]:
    """
    the function creates a csv file

    Parameters
    ----------
    folder_path (str): The path to the main folder containing subfolders.
    subfolder_paths (List[str]): List of paths to subfolders within the main folder.
    annotation_file_path (str): The path where the CSV annotation file will be created.

    Returns:
    Tuple[str, str, str]: A tuple containing absolute path, relative path, and class name.
    """
    try:
        with open(annotation_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([ABSOLUTE_PATH_COLUMN, RELATIVE_PATH_COLUMN, CLASS_NAME_COLUMN])

            for subfolder_path in subfolder_paths:
                class_name = os.path.basename(subfolder_path)

                for filename in os.listdir(os.path.join(folder_path, subfolder_path)):
                    absolute_path = os.path.join(folder_path, subfolder_path, filename)
                    relative_path = os.path.join(subfolder_path, filename)
                    csv_writer.writerow([absolute_path, relative_path, class_name])

        logging.info(f"The file with the annotation has been created: {annotation_file_path}")
        return absolute_path, relative_path, class_name
    except Exception as e:
        logging.exception(f"Error in creating an annotation file: {e}")
        return '', '', ''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create annotation file for the dataset')
    parser.add_argument('folder_path', type=str, default='dataset', help='Path to the dataset directory')
    parser.add_argument('subfolder_paths', nargs='+', type=str, help='List of subfolder paths')
    parser.add_argument('annotation_file', type=str, default='annotation.csv', help='Path for the annotation file')

    args = parser.parse_args()
    create_annotation_file(args.folder_path, args.subfolder_paths, args.annotation_file)