
import os
import csv
import shutil
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_annotation_file(dataset_directory: str, output_file: str, include_relative_path: bool = False) -> None:
    """Create annotation file for the dataset."""
    try:
        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Absolute Path', 'Relative Path', 'Class Label'])

            for root, dirs, files in os.walk(dataset_directory):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, start=dataset_directory) if include_relative_path else None
                    class_label = os.path.basename(root)
                    csvwriter.writerow([abs_path, rel_path, class_label])

        logger.info(f"Annotation file '{output_file}' created successfully.")
    except Exception as e:
        logger.exception(f"Error creating annotation file: {e}")
