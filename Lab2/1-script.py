import csv
import os
from typing import Generator


def get_annotation(dir='dataset') -> Generator[list, None, None]:
    """
    the function creates list of lists consisting of three elements:
    relative path, absolute path and class label for each file.
    ----------
    dir : str
    """
    for star in range(1, 6):
        directory = os.path.join(dir, str(star))
        files = [file for file in os.listdir(
            directory) if os.path.isfile(f'{directory}/{file}')]
        cnt_files = len(files)
        for file in range(1, cnt_files + 1):
            relative_path = os.path.join(
                directory, f'{str(file).zfill(4)}.txt')
            absolute_path = os.path.abspath(relative_path)
            yield [relative_path, absolute_path, star]


def write_csv(path='reviews') -> None:
    """
    the function writes list elements to a csv file.
    ----------
    path : str
    """
    with open(path, 'w', newline='') as file:
        csv.writer(file).writerows(list(get_annotation()))


if __name__ == '__main__':
    write_csv()
