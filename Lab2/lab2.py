import os
import csv
import logging


def create_annotation_file(dataset_path, output_file):
    """
    Функция создания аннотации

    """
    try:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['absolute_path', 'relative_path', 'class_label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(('.txt')):
                        absolute_path = os.path.join(root, file)
                        relative_path = os.path.relpath(absolute_path, dataset_path)
                        class_label = os.path.basename(root)

                        writer.writerow({'absolute_path': 'Абсолютный путь : '+absolute_path,
                                        'relative_path': ' Относительный путь: '+relative_path,
                                        'class_label': ' Тип рецензии: '+class_label})
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}", exc_info=True)
    return None


if __name__ == "__main__":
    dataset_path = r"C:\Users\Ceh9\PycharmProjects\pythonProject"
    output_file = "annotation.csv"
    create_annotation_file(dataset_path, output_file)
