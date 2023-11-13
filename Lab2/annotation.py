import csv
import os
import shutil
import random
import logging
from typing import Generator


def mkdir(path: str) -> None:
    """
    func handles folder creation using exceptions
    Parameters.
    ----------
    path : str
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as err:
        logging.error(f"{err}", exc_info=True)

def copy_with_rand_num(old_dir='dataset',new_dir='dataset_with_rand_num') -> Generator[list, None, None]:
    """
    the function copy each file to new directory.
    ----------
    old_dir : str
    new_dir : str
    """
    mkdir(new_dir)
    for star in range(1, 6):
        dir = os.path.join(old_dir, f'{star}')
        for file in os.listdir(dir):
            path_to_file=f'{str(random.randrange(10000)).zfill(4)}.txt'
            while os.path.isfile(os.path.join(new_dir, path_to_file)):#replace number, if file exists
                path_to_file=f'{str(random.randrange(10000)).zfill(4)}.txt'
            shutil.copy(os.path.join(dir, file), os.path.join(new_dir, path_to_file))
            relative_path = os.path.join(new_dir, path_to_file)
            absolute_path = os.path.abspath(relative_path)
            yield [absolute_path, relative_path, star]

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
        for file in range(1, len(files) + 1):
            relative_path = os.path.join(
                directory, f'{str(file).zfill(4)}.txt')
            absolute_path = os.path.abspath(relative_path)
            yield [absolute_path,relative_path, star]



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


def get_annotation_new_dir(old_dir='dataset',new_dir='new_dataset') -> Generator[list, None, None]:
    """
    the function creates list of lists consisting of three elements:
    relative path, absolute path and class label for each file.
    ----------
    dir : str
    """
    copy_to_dir(old_dir,new_dir)
    for file in os.listdir(new_dir):
        relative_path = os.path.join(new_dir,f'{file}')
        absolute_path = os.path.abspath(relative_path)
        yield [absolute_path, relative_path, file[0]]

def write_csv(path_to_csv='reviews.csv',label_of_annotation=0,new_dir='new_dataset',old_dir='dataset') -> str:
    """
    the function writes list elements to a csv file.
    return path to csv file.
    ----------
    path : str
    """
    list=[]
    if label_of_annotation==0:
        list=get_annotation(old_dir)
    elif label_of_annotation==1:
        list=get_annotation_new_dir(old_dir,new_dir)
    elif label_of_annotation==2:
        list=copy_with_rand_num(old_dir,new_dir)
    else:
        raise Exception("Incorrect mode")
    try:
        with open(path_to_csv, 'w', newline='') as file:
            csv.writer(file).writerows(list)
    except Exception as ex:
        logging.error(f"Error of writing row in csv: {ex}")
    return path_to_csv


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=os.path.join("py_log.log"), filemode="w")
