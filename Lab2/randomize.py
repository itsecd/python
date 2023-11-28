import os
import logging
import shutil
import random
import json
import csv_build


logging.basicConfig(level=logging.INFO)


def make_rand_list(top: int) -> list:
    """Creates a list filled with random numbers from 0 to {top}"""
    rand_list = []
    for i in range(0, top):
        rand_list.append(i)
    random.shuffle(rand_list)
    return rand_list


def randomize_dataset(dataset: str, rand_dataset: str, classes: list, size: int) -> list:
    """Creates a {rand_dataset} folder with files from {dataset} with randomized names. 
    Returns a list of paths to them associated with respective {classes}
    Requires that {rand_dataset} folder is missing or empty for proper use"""
    path_list = list()
    rand_list = make_rand_list(size)
    if not os.path.exists(os.path.join(rand_dataset)):
        os.mkdir(os.path.join(rand_dataset))
    cnt = 0
    for cls in classes:
        files_count = len(os.listdir(os.path.join(dataset, cls)))
        for i in range(files_count-10):
            normal = os.path.abspath(os.path.join(dataset, cls, f'{i:04}.txt'))
            randomized = os.path.abspath(os.path.join(rand_dataset, f'{rand_list[cnt]:04}.txt'))
            shutil.copy(normal, randomized)
            path_set = [
                [randomized,
                 os.path.relpath(randomized),
                 cls,]
            ]
            path_list += path_set
            cnt += 1
    return path_list


if __name__ == '__main__':
    with open(os.path.join('Lab2', 'settings.json'), 'r') as settings:
        settings = json.load(settings)
    rand_pathlist = randomize_dataset(os.path.join(settings['directory'], settings['dataset_folder']), 
                                      os.path.join(settings['directory'], settings['random_dataset']), 
                                      settings['classes'],  
                                      settings['default_size'])
    csv_build.write_into_file((os.path.join(settings['csv_folder'], settings['randomized_csv'])), rand_pathlist)