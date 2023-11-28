import os
import logging
import shutil
import random
import json
import create_annotation
import csv

def make_random_list(top: int) -> list:
    """Creates a list filled with random numbers from 0 to {top}"""
    rand_list = []
    for i in range(0, top):
        rand_list.append(i)
    random.shuffle(rand_list)
    return rand_list

def random_dataset(dataset: str, random_dataset: str, size: int, classes: list, csv_file_name: str) -> list:
    """Создает папку, где файлы из random_dataset получают случайные имена."""
    random_idx = make_random_list(size)

    path_list = list()
    if not os.path.exists(os.path.join(random_dataset)):
        os.mkdir(os.path.join(random_dataset))

    count = 0
    for cls in classes:
        files_count = len(os.listdir(os.path.join(dataset, cls)))
        for i in range(files_count):
            source_path = os.path.abspath(os.path.join(dataset, cls, f'{i:04}.txt'))
            target_path = os.path.abspath(os.path.join(random_dataset, f'{random_idx[count]:04}.txt'))
            shutil.copy(source_path, target_path)
            path_set = [
                [target_path,
                os.path.relpath(target_path),cls
                ]
            ]
            path_list += path_set
            count+=1

            csv_file_path = os.path.join(os.getcwd(), csv_file_name)
            with open(csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Absolute Path', 'Relative Path', 'Class'])
                csv_writer.writerows(path_list)


if __name__ == '__main__':
    with open(os.path.join('Lab2', 'settings.json'), 'r') as settings_file:
        settings = json.load(settings_file)

    random_dataset(settings['main_dataset'],settings['dataset_random'],settings['default_number'],settings['classes'],settings['random_csv'])
    