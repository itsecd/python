import os
from pathlib import Path
import csv
import shutil
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_annotation_file(dataset_directory: str, output_file: str, include_relative_path: bool = False) -> None:
    """Create annotation file for the dataset."""
    try:
        paths = []
        for root, dirs, files in os.walk(dataset_directory):
            for file in files:
                abs_path = Path(root) / file
                rel_path = abs_path.relative_to(dataset_directory) if include_relative_path else None
                class_label = Path(root).name
                paths.append((abs_path, rel_path, class_label))

        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Absolute Path', 'Relative Path', 'Class Label'])
            csvwriter.writerows(paths)

        logger.info(f"Annotation file '{output_file}' created successfully.")
    except Exception as e:
        logger.exception(f"Error creating annotation file: {e}")

def copy_dataset_with_index(dataset_directory: str, output_directory: str, include_relative_path: bool = False) -> None:
    """Copy dataset with modified file names and create annotation file."""
    annotation_file = Path(output_directory) / "annotation_class_index.csv"

    try:
        paths = []
        for root, dirs, files in os.walk(dataset_directory):
            for i, file in enumerate(files):
                class_label = Path(root).name
                new_name = f"{class_label}_{i:04d}.jpg"
                source_path = Path(root) / file
                destination_path = Path(output_directory) / new_name
                paths.append((destination_path, destination_path.relative_to(output_directory) if include_relative_path else None, class_label))
                shutil.copyfile(source_path, destination_path)

        with open(annotation_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Absolute Path', 'Relative Path', 'Class Label'])
            csvwriter.writerows(paths)

        logger.info(f"Dataset with class index copied successfully. Annotation file: '{annotation_file}'.")
    except Exception as e:
        logger.exception(f"Error copying dataset with class index: {e}")

def copy_dataset_with_random_numbers(dataset_directory: str, output_directory: str, include_relative_path: bool = False) -> None:
    """Copy dataset with random file names and create annotation file."""
    annotation_file = Path(output_directory) / "annotation_random_numbers.csv"

    try:
        paths = []
        for root, dirs, files in os.walk(dataset_directory):
            for file in files:
                class_label = Path(root).name
                random_number = random.randint(0, 10000)
                new_name = f"{random_number}.jpg"
                source_path = Path(root) / file
                destination_path = Path(output_directory) / new_name
                paths.append((destination_path, destination_path.relative_to(output_directory) if include_relative_path else None, class_label))
                shutil.copyfile(source_path, destination_path)

        with open(annotation_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Absolute Path', 'Relative Path', 'Class Label'])
            csvwriter.writerows(paths)

        logger.info(f"Dataset with random numbers copied successfully. Annotation file: '{annotation_file}'.")
    except Exception as e:
        logger.exception(f"Error copying dataset with random numbers: {e}")

if __name__ == "__main__":
    dataset_dir = "dataset"
    
    create_annotation_file(dataset_dir, "annotation_absolute_path.csv")
    copy_dataset_with_index(dataset_dir, "dataset_with_class_index", include_relative_path=True)
    copy_dataset_with_random_numbers
