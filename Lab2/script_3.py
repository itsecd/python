import json
import os
import random
import logging
import shutil
from script_1 import create_csv, write_in_csv
from get_way import get_path_random
from iterator import Iterator

def make_new_fold(name_csv : str, img_classes : list, directory : str) -> None:
    try:
        create_csv(name_csv)
        for img_class in img_classes:
            iterator = Iterator(f'Lab2/{sj["normal"]}.csv', img_class)
            for element in iterator:
                if element != None and os.path.isfile(str(element)):
                    _random = random.randint(0, 10000)
                    shutil.copyfile(str(element), get_path_random(img_class, _random))
                    write_in_csv(name_csv, img_class, get_path_random(img_class, _random))
                else: 
                    del iterator
                    break
    except Exception as ex:
        logging.error(f"Error of copy img : {iterator.counter-1} | {ex}")


if __name__ == '__main__':
    with open(os.path.join("Lab1", "input_data.json"), 'r') as fjson:
        fj = json.load(fjson)

    with open(os.path.join("Lab2", "src_csv.json"), 'r') as srcjson:
        sj = json.load(srcjson)

    if not os.path.isdir(os.path.join(fj["main_folder"], sj["random"])):
        os.mkdir(os.path.join(fj["main_folder"], sj["random"]))

    logging.basicConfig(level=logging.INFO)
    make_new_fold(os.path.join("Lab2", sj["random"]), fj["objects"], fj["main_folder"])
