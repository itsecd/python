import os
import csv
import logging
import json

logging.basicConfig(filename='annotation.log', level=logging.ERROR)

def create_annotation_file(dataset_path, output_file):
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Абсолютный путь'.ljust(100), 'Относительный путь'.ljust(100), 'Класс'.ljust(30)])

            for file_dataset, subfolders, files_list in os.walk(dataset_path):
                for file_name in files_list:
                    if file_name.endswith('.txt'):
                        absolute_path = os.path.abspath(os.path.join(file_dataset, file_name)).ljust(100)
                        relative_path = os.path.relpath(absolute_path, os.path.dirname(__file__)).ljust(100)
                        class_name = os.path.basename(file_dataset).ljust(30)
                        csv_writer.writerow([absolute_path, relative_path, class_name])
    except Exception as e:
        logging.error(f"Произошла ошибка в create_annotation_file: {e}", exc_info=True)

if __name__ == "__main__":
    with open(os.path.join('Lab2', 'settings.json'), 'r') as settings:
        settings = json.load(settings)
    create_annotation_file(settings['normal'], settings['normal_csv'])
