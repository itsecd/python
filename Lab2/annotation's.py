
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

def copy_dataset_with_index(dataset_directory: str, output_directory: str, include_relative_path: bool = False) -> None:
    """Copy dataset with modified file names and create annotation file."""
    annotation_file = os.path.join(output_directory, "annotation_class_index.csv")
    
    try:
        with open(annotation_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Absolute Path', 'Relative Path', 'Class Label'])

            for root, dirs, files in os.walk(dataset_directory):
                for i, file in enumerate(files):
                    class_label = os.path.basename(root)
                    new_name = f"{class_label}_{i:04d}.jpg"
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(output_directory, new_name)
                    shutil.copyfile(source_path, destination_path)
                    abs_path = os.path.abspath(destination_path)
                    rel_path = os.path.relpath(abs_path, start=output_directory) if include_relative_path else None
                    csvwriter.writerow([abs_path, rel_path, class_label])

        logger.info(f"Dataset with class index copied successfully. Annotation file: '{annotation_file}'.")
    except Exception as e:
        logger.exception(f"Error copying dataset with class index: {e}")

def copy_dataset_with_random_numbers(dataset_directory: str, output_directory: str, include_relative_path: bool = False) -> None:
    """Copy dataset with random file names and create annotation file."""
    annotation_file = os.path.join(output_directory, "annotation_random_numbers.csv")
    
    try:
        with open(annotation_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Absolute Path', 'Relative Path', 'Class Label'])

            for root, dirs, files in os.walk(dataset_directory):
                for file in files:
                    class_label = os.path.basename(root)
                    random_number = random.randint(0, 10000)
                    new_name = f"{random_number}.jpg"
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(output_directory, new_name)
                    shutil.copyfile(source_path, destination_path)
                    abs_path = os.path.abspath(destination_path)
                    rel_path = os.path.relpath(abs_path, start=output_directory) if include_relative_path else None
                    csvwriter.writerow([abs_path, rel_path, class_label])

        logger.info(f"Dataset with random numbers copied successfully. Annotation file: '{annotation_file}'.")
    except Exception as e:
        logger.exception(f"Error copying dataset with random numbers: {e}")

if __name__ == "__main__":
    dataset_dir = "dataset"
    
    create_annotation_file(dataset_dir, "annotation_absolute_path.csv")
    copy_dataset_with_index(dataset_dir, "dataset_with_class_index", include_relative_path=True)
    copy_dataset_with_random_numbers(dataset_dir, "dataset_with_random_numbers", include_relative_path=True)
