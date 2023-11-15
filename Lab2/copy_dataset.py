import shutil
import os
import csv
import json


def copy_dataset_with_annotation(dataset_path: str, destination_path: str, annotation_file_path: str) -> None:
    """Copies files from dataset to copied_dataset with renaming by format {class}_{number}.txt."""
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    with open(annotation_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Original Path', 'New Path', 'Class'])

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                class_name = os.path.basename(root)
                original_path = os.path.join(root, file)
                new_filename = f"{class_name}_{file}"
                new_path = os.path.join(destination_path, new_filename)

                shutil.copy(original_path, new_path)

                csv_writer.writerow([original_path, new_path, class_name])

if __name__ == "__main__":
    settings_file_path = os.path.join('Lab2', 'settings.json')

    if not os.path.exists(settings_file_path):
        print(f"Error: Settings file '{settings_file_path}' not found.")
    else:
        with open(settings_file_path, 'r') as settings_file:
            settings = json.load(settings_file)

        copy_dataset_with_annotation(settings['dataset_folder'], settings['copied_dataset'], settings['copied_csv'])
