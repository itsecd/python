import json
import os
import random
import logging
import shutil
from script_1 import create_csv, write_in_csv

def make_new_fold(name_csv : str, img_classes : list, directory : str) -> None:
    try:
        create_csv(name_csv)
        for img_class in img_classes:
            number_of_img = len(os.listdir(os.path.join(directory, img_class)))
            for img in range(number_of_img):
                _random = random.randint(0, 10000)
                shutil.copyfile(os.path.join(directory, img_class, f"{img:04}.jpg"), os.path.join(directory, sj["random"], f"{_random}.jpg"))
                # os.remove(os.path.join(directory, img_class, f"{img:04}.jpg"))

                write_in_csv(name_csv, img_class, os.path.join(directory, f"{_random}.jpg"))
            
    except Exception as ex:
        logging.error(f"Error of copy img : {img} | {ex}")


if __name__ == '__main__':
    with open(os.path.join("Lab1", "input_data.json"), 'r') as fjson:
        fj = json.load(fjson)

    with open("Lab2/src_csv.json", "r") as srcjson:
        sj = json.load(srcjson)

    logging.basicConfig(level=logging.INFO)
    make_new_fold(os.path.join("Lab2", sj["random"]), fj["objects"], fj["main_folder"])
