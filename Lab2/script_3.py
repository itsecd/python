import csv
import json
import os
import time
import random
import logging
import shutil
from script_1 import create_csv

def make_new_folder(name_csv : str, img_class : str, directory : str) -> None:
    try:
        create_csv(name_csv)
        number_of_img = len(os.listdir(os.path.join(directory, img_class)))
        for img in range(number_of_img):
            shutil.copyfile(os.path.join(directory, f"{random.randint(0, 10000)}.jpg"), os.path.join(directory, f"{img_class}_{img:04}.jpg"))
            # os.remove(os.path.join(directory, img_class, f"{img:04}.jpg"))

            # write_in_csv(name_csv, img_class, os.path.join(directory, f"{img_class}_{img:04}.jpg"))
            
    except Exception as ex:
        logging.error(f"Error of copy img : {img} | {ex}")

if __name__ == '__main__':
    with open(os.path.join("Lab1", "input_data.json"), 'r') as fjson:
        fj = json.load(fjson)

    logging.basicConfig(level=logging.INFO)