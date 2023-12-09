import os
import shutil
import csv
import logging
from create_annotation import create_annotation_file

def copy_and_rename_dataset(input_path, output_path):
    """
    Функция копирывания и изменения имени датасета
    """
    try:
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith(('.txt')):
                    class_label = os.path.basename(root)
                    new_file_name = f"{class_label}_{file}"

                    input_file_path = os.path.join(root, file)
                    output_file_path = os.path.join(output_path, class_label, new_file_name)

                    os.makedirs(os.path.join(output_path, class_label), exist_ok=True)
                    shutil.copy(input_file_path, output_file_path)
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}", exc_info=True)
    return None


if __name__ == "__main__":
    input_dataset_path = r"C:\Users\Ceh9\PycharmProjects\pythonProject"
    output_dataset_path = r"C:\Users\Ceh9\PycharmProjects\pythonProject\Lab2"
    output_file = "_annotation.csv"
    copy_and_rename_dataset(input_dataset_path, output_dataset_path)
    create_annotation_file(output_dataset_path, output_file)