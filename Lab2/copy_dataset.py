import os
import shutil
import csv
import json
import logging
import random
from create_annotation import create_annotation_file

def copy_and_rename_dataset(input_path: str, output_path: str, name_of_output_file: str, random_or_not :bool) -> None:
    """
    Функция копирования и изменения имени датасета рандомом и нет
    """
    try:
        if not os.path.exists(name_of_output_file):
            os.mkdir(name_of_output_file)
            output_path = os.path.join(output_path, "/", name_of_output_file)
        for root, dirs, files in os.walk(input_path):
            for file in files:
                class_label = os.path.basename(root)
                if (file.endswith(('.txt'))):
                        if (random_or_not):
                            random_number = random.randint(0, 10000)
                            new_file_name = f"{random_number}.txt"
                        else:
                            new_file_name = f"{class_label}_{file}.txt"
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_path, class_label, new_file_name)
                os.makedirs(os.path.join(output_path, class_label), exist_ok=True)
                shutil.copy(input_file_path, output_file_path)

    except Exception as e:
        logging.error(f"Произошла ошибка: {e}", exc_info=True)


if __name__ == "__main__":
    with open("C://Users/Ceh9/PycharmProjects/pythonProject/Lab2/options.json", "r") as options_file:
        options = json.load(options_file)
        copy_and_rename_dataset(options["dataset_path"], options["output_dataset_path"], options["name_of_output_file"], options["random_or_not"][0])
        create_annotation_file(options["output_dataset_path"], options["output_file_annotation"])
