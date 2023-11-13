import csv
import os
import shutil
import random
import logging
from typing import Generator
from enum import Enum


class AnnotationLabel(Enum):
    DEFAULT = 0
    NEWDIR = 1
    RAND = 2


def makedir(path: str) -> None:
    """
    func handles folder creation using exceptions
    Parameters.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as err:
        logging.error(f"{err}", exc_info=True)


def get_annotation(dir='dataset', new_dir='new_dataset', label_annotation=AnnotationLabel.DEFAULT) -> Generator[
    list, None, None]:
    """
    the function creates list of lists consisting of three elements:
    relative path, absolute path and class label for each file.
    """
    if (label_annotation != 0):
        makedir(new_dir)
    for star in range(1, 6):
        dir = os.path.join(dir, f'{star}')
        files = [file for file in os.listdir(dir) if os.path.isfile(f'{dir}/{file}')]
        for file in files:
            match label_annotation:
                case AnnotationLabel.DEFAULT:
                    relative_path = os.path.join(dir, f'{str(file).zfill(4)}.txt')
                case AnnotationLabel.NEWDIR:
                    shutil.copy(os.path.join(dir, file), os.path.join(new_dir, f'{star}_{file}'))
                    relative_path = os.path.join(new_dir, f'{file}')
                case AnnotationLabel.RAND:
                    path_to_file = f'{str(random.randrange(10000)).zfill(4)}.txt'
                    while os.path.isfile(os.path.join(new_dir, path_to_file)):  # replace number, if file exists
                        path_to_file = f'{str(random.randrange(10000)).zfill(4)}.txt'
                    shutil.copy(os.path.join(dir, file), os.path.join(new_dir, path_to_file))
                    relative_path = os.path.join(new_dir, path_to_file)
                case _:
                    raise Exception("Incorrect mode")
            absolute_path = os.path.abspath(relative_path)
            yield [absolute_path, relative_path, star]


def write_csv(path_to_csv='reviews.csv', label_of_annotation=AnnotationLabel.DEFAULT, new_dir='new_dataset',
              old_dir='dataset') -> str:
    """
    the function writes list elements to a csv file.
    return path to csv file.
    """
    list = get_annotation(old_dir, new_dir, label_of_annotation)
    try:
        with open(path_to_csv, 'w', newline='') as file:
            csv.writer(file).writerows(list)
    except Exception as ex:
        logging.error(f"Error of writing row in csv: {ex}")
    return path_to_csv


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=os.path.join("py_log.log"), filemode="w")
