import os
import csv
import json


def generate_annotation_file(dataset_path: str, annotation_file_path: str) -> None:
    with open(annotation_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Absolute Path', 'Relative Path', 'Class'])

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                class_name = os.path.basename(os.path.dirname(os.path.join(root, file)))
                relative_path = os.path.relpath(os.path.join(root, file), dataset_path)
                absolute_path = os.path.abspath(os.path.join(root, file))
                csv_writer.writerow([absolute_path, relative_path, class_name])

if __name__ == "__main__":
    with open(os.path.join('Lab2', 'settings.json'), 'r') as settings:
        settings = json.load(settings)
    generate_annotation_file(settings['dataset_folder'], settings['default_csv'])
