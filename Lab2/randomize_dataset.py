import os
import logging
import shutil
import random
import csv
import json


logging.basicConfig(level=logging.INFO)


def make_random_list(top: int) -> list:
    """Creates a list filled with random numbers from 0 to {top}"""
    """Creates a list filled with random numbers from 0 to {top}"""
    rand_list = []
    for i in range(0, top):
        rand_list.append(i)
    random.shuffle(rand_list)
    return rand_list


def randomize_dataset_with_annotation(dataset: str, path: str, rand_dataset: str, classes: list, size: int) -> list:
    path_list = list()
    rand_list = make_random_list(size)
    if not os.path.exists(os.path.join(rand_dataset)):
        os.mkdir(os.path.join(rand_dataset))
    cnt = 0
    for cls in classes:
        files_count = len(os.listdir(os.path.join(dataset, cls)))
        for i in range(files_count):
            normal = os.path.abspath(os.path.join(dataset, cls, f'{i:04}.jpg'))
            randomized = os.path.abspath(os.path.join(rand_dataset, f'{rand_list[cnt]:04}.jpg'))
            shutil.copy(normal, randomized)
            path_set = [[randomized, os.path.relpath(randomized),cls,] ]
            path_list += path_set
            cnt += 1

            csv_file_path = os.path.join(os.getcwd(), path)
            with open(csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Absolute Path', 'Relative Path', 'Class'])
                csv_writer.writerows(path_list)

if __name__ == "__main__":
    with open(os.path.join('Lab2', 'settings.json'), 'r') as settings:
        settings = json.load(settings)
    randomize_dataset_with_annotation( settings['dataset_folder'], settings['randomized_csv'], settings['randomized_dataset'], settings['classes'],  settings['default_size'])
    
