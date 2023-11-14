import os
import csv
import logging

logging.basicConfig(filename='annotation.log', level=logging.ERROR)

def create_annotation_file(dataset_path, output_file):
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Абсолютный путь'.ljust(100), 'Относительный путь'.ljust(100), 'Класс'.ljust(30)])

            for folder_path, subfolders, files_list in os.walk(dataset_path):
                for file_name in files_list:
                    if file_name.endswith('.txt'):
                        absolute_path = os.path.abspath(os.path.join(folder_path, file_name)).ljust(100)
                        relative_path = os.path.relpath(absolute_path, os.path.dirname(__file__)).ljust(100)
                        class_name = os.path.basename(folder_path).ljust(30)
                        csv_writer.writerow([absolute_path, relative_path, class_name])
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    dataset_folder = "C:/Users/ksush/OneDrive/Рабочий стол/python-v8/dataset"
    annotation_file = "annotation.csv"

    create_annotation_file(dataset_folder, annotation_file)
