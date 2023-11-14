import os
import csv
import json
import logging

logging.basicConfig(level=logging.INFO)

# Функция для получения списка файлов в папке
def list_files_in_directory(folder):
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg"):
                file_list.append(os.path.join(root, file))
    return file_list

# Формирование абсолютного пути, относительного пути и класса для каждого файла
def generate_annotation(main_folder, queries):
    file_list = list_files_in_directory(main_folder)
    annotations = []
    for query in queries:
         for file in file_list:
            absolute_path_from_project = os.path.abspath(file)
            relative_path_from_project = os.path.relpath(file, main_folder)
            annotations.append([absolute_path_from_project, relative_path_from_project, query])
    return annotations


def create_annotation_file(annotation_file: str) -> None:
    """Функция принимает имя файла и создает файл, если его не существует"""
    try:
        if not os.path.exists(annotation_file):
             with open(f"{annotation_file}.csv", "w", newline='') as file:
                pass
    except Exception as ex:
        logging.error(f"Не удалось создать файл: {ex}\n")

# Запись аннотаций в CSV-файл
def write_annotation_to_csv(annotations, annotation_file):
    create_annotation_file(annotation_file)
    with open(f"{annotation_file}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(annotations)


if __name__ == "__main__":
    with open("Lab2/options.json", "r") as options_file:
        options = json.load(options_file)
        annotations = generate_annotation(options["main_folder"], options["queries"])
        write_annotation_to_csv(annotations, options["annotation_file"])