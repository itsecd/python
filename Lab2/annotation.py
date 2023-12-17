import argparse
import csv
import logging
import os

def create_annotation_file(folder_path: str, subfolder_paths: list, annotation_file_path: str):
    """
    the function creates a csv file

    Parameters
    ----------
    folder_path : str
    subfolder_paths : list
    annotation_file_path : str
    """
    try:
        with open(annotation_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['The absolute path', 'relative path', 'the text name of the class'])

            for subfolder_path in subfolder_paths:
                class_name = os.path.basename(subfolder_path)

                for filename in os.listdir(os.path.join(folder_path, subfolder_path)):
                    absolute_path = os.path.join(folder_path, subfolder_path, filename)
                    relative_path = os.path.join(subfolder_path, filename)
                    csv_writer.writerow([absolute_path, relative_path, class_name])

        logging.info(f"The file with the annotation has been created: {annotation_file_path}")
    except Exception as e:
        logging.exception(f"Error in creating an annotation file: {e}")
