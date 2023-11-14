import shutil
import os
import csv


def copy_dataset_with_prefix(dataset_path: str, destination_path: str) -> None:
    with open('annotation.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            _, relative_path, class_name = row
            source_path = os.path.join(dataset_path, relative_path)
            destination_filename = f"{class_name}_{os.path.basename(source_path)}"
            destination_path_class = os.path.join(destination_path, destination_filename)
            shutil.copy(source_path, destination_path_class)

if __name__ == "__main__":
    copy_dataset_with_prefix('dataset', 'dataset_with_prefix')
