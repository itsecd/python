import os
import csv
import json
import logging
FIELDNAMES = ['absolute_path', 'relative_path', 'class_label']


def create_annotation_file(dataset_path: str, output_file: str) -> None:
    """
    Функция создания аннотации

    """
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            writer.writeheader()
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(('.txt')):
                        absolute_path = os.path.join(root, file)
                        relative_path = os.path.relpath(absolute_path, dataset_path)
                        class_label = os.path.basename(root)

                        writer.writerow({'absolute_path': absolute_path,
                                        'relative_path': relative_path,
                                        'class_label': class_label})
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}", exc_info=True)



if __name__ == "__main__":
    with open("C://Users/Ceh9/PycharmProjects/pythonProject/Lab2/options.json", "r") as options_file:
        options = json.load(options_file)
        create_annotation_file(options["dataset_path"], options["output_file"])
