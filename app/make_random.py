import os
import logging
import shutil
import random
from usefull_functions import create_csv, write_in_csv
from get_way import get_path_normal, get_path_random

def make_new_fold(name_csv : str, img_classes : list, directory : str) -> None:
    """
    Make new csv file

    Parameters
    ----------
    name_csv : str
        Name of csv file
    img_classes : str
        List of objects
    directory : str
        Path to dataset
    """
    try:
        create_csv(name_csv)
        for img_class in img_classes:
            number_of_img = len(os.listdir(os.path.join(directory, img_class)))
            for img in range(number_of_img):
                _random = random.randint(0, 10000)
                shutil.copyfile(get_path_normal(img_class, img), get_path_random(img_class, _random))
                write_in_csv(name_csv, img_class, get_path_random(img_class, _random))
    except Exception as ex:
        logging.error(f"Error of copy img : {img} | {ex}")