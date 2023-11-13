import csv
import os
import json
import logging


logging.basicConfig(level=logging.INFO)


def make_csv(name: str) -> None:

    try:
        if not os.path.exists(name):
            with open(f'{name}.csv', 'a') as file:
                csv.writer(file, lineterminator='\n')
    except Exception as exc:
        logging.error(f'Failed to create file: {exc.message}: {exc.args}\n')


def make_list(dir: str, classes: str) -> list:

    review_list = list()
    for c in classes:
        files_count = len(os.listdir(os.path.join(dir, c)))
        for i in range(files_count):
            path_set = [
                [os.path.abspath(os.path.join(dir, c, f'{i:04}.txt')),
                 os.path.join(dir, c, f'{i:04}.txt'),
                 c,]
            ]
            review_list += path_set
    return review_list


def write_into_file(name: str, review_list: list) -> None:

    try:
        make_csv(name)
        for review in review_list:
            with open(f'{name}.csv', 'a') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerow(review)
    except Exception as exc:
        logging.error(f'Failed to write data: {exc.message}: {exc.args}\n')