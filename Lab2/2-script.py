import shutil
import os
import csv
from typing import Generator


def copy_to_dir(old_dir='dataset',new_dir='new_dataset'):
    """
    the function copy each file to new directory.
    ----------
    old_dir : str
    new_dir : str
    """
    os.mkdir(new_dir)
    for star in range(1, 6):
        dir = os.path.join(old_dir, f'{star}')
        for file in os.listdir(dir):
            shutil.copy(os.path.join(dir, file), os.path.join(new_dir, f'{star}_{file}'))


def get_annotation(new_dir='new_dataset') -> Generator[list, None, None]:
    """
    the function creates list of lists consisting of three elements:
    relative path, absolute path and class label for each file.
    ----------
    dir : str
    """
    for file in os.listdir(new_dir):
        relative_path = os.path.join(new_dir,f'{file}')
        absolute_path = os.path.abspath(relative_path)
        yield [relative_path, absolute_path, file[0]]


def write_csv(path='new_reviews.csv') -> None:
    """
    the function writes list elements to a csv file.
    ----------
    path : str
    """
    with open(path, 'w', newline='') as file:
        csv.writer(file).writerows(list(get_annotation()))


if __name__ == '__main__':
    copy_to_dir()
    write_csv()
