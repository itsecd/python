import os
import shutil
import csv
import logging
import random
from create_annotation import create_annotation_file

def copy_and_rename_dataset(input_path, output_path, RANDOM:bool):
    """
    Функция копирывания и изменения имени датасета рандомом и нет
    """
    try:
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if (RANDOM):
                    if file.endswith(('.txt')):
                            random_number = random.randint(0, 10000)
                            class_label = os.path.basename(root)
                            new_file_name = f"{random_number}_{file}"
                            input_file_path = os.path.join(root, file)
                            output_file_path = os.path.join(output_path, class_label, new_file_name)
                            os.makedirs(os.path.join(output_path, class_label), exist_ok=True)
                            shutil.copy(input_file_path, output_file_path)
                else:
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
    RANDOM = True
    input_dataset_path = r"C:\Users\Ceh9\PycharmProjects\pythonProject\Lab1\dataset"
    output_dataset_path = r"C:\Users\Ceh9\PycharmProjects\pythonProject\Lab2\_dataset"
    output_file = "_annotation.csv"
    copy_and_rename_dataset(input_dataset_path, output_dataset_path, RANDOM)
    create_annotation_file(output_dataset_path, output_file)