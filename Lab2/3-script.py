import shutil
import os
import csv
from typing import Generator
import random

def copy_with_rand_num(old_dir='dataset',new_dir='dataset_with_rand_num') -> Generator[list, None, None]:
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
            path_to_file=f'{str(random.randrange(10000)).zfill(4)}.txt'
            while os.path.isfile(os.path.join(new_dir, path_to_file)):#replace number, if file exists
                path_to_file=f'{str(random.randrange(10000)).zfill(4)}.txt'
            shutil.copy(os.path.join(dir, file), os.path.join(new_dir, path_to_file))
            relative_path = os.path.join(new_dir, path_to_file)
            absolute_path = os.path.abspath(relative_path)
            yield [relative_path, absolute_path, star]



def write_csv(path='new_reviews_w_rand_num') -> None:
    """
    the function writes list elements to a csv file.
    ----------
    path : str
    """
    with open(path, 'w', newline='') as file:
        csv.writer(file).writerows(list(copy_with_rand_num()))

if __name__ == '__main__':
    write_csv()