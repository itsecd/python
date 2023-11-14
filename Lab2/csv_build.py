import csv
import os
import json
import logging


logging.basicConfig(level=logging.INFO)


def make_csv(name: str) -> None:
    """Creates a .csv file named {str}"""
    try:
        if not os.path.exists(name):
            with open(f'{name}', 'a') as file:
                csv.writer(file, lineterminator='\n')
    except Exception as exc:
        logging.error(f'Failed to create file: {name}\n{exc.args}\n')


def make_pathlist(dir: str, classes: list) -> list:
    """Creates a list of paths to reviews in .txt files inside {classes[i]} folders in {dir} folder"""
    review_list = list()
    for cls in classes:
        files_count = len(os.listdir(os.path.join('Lab2', dir, cls)))
        for i in range(files_count):
            path_set = [
                [os.path.abspath(os.path.join(dir, cls, f'{i:04}.txt')),
                 os.path.join(dir, cls, f'{i:04}.txt'),
                 cls,]
            ]
            review_list += path_set
    return review_list


def write_into_file(name: str, review_list: list) -> None:
    """Writes a list of reviews {review_list} into a {name}.csv file"""
    try:
        make_csv(name)
        for review in review_list:
            with open(f'{name}', 'a') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerow(review)
    except Exception as exc:
        logging.error(f'Failed to write data: {exc.args}\n')


if __name__ == '__main__':
    with open(os.path.join('Lab2', 'settings.json'), 'r') as settings:
        settings = json.load(settings)
    review_list = make_pathlist(settings['dataset_folder'], settings['classes'])
    write_into_file(os.path.join(settings['csv_folder'], settings['pathfile_csv']), review_list)