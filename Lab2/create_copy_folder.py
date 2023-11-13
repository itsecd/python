import os
import logging
import shutil
import json
import create_annotation

logging.basicConfig(level=logging.INFO)


def copy_folder(old_dir: str,new_dir: str, classes: list, csv_name:str) -> None:
    """This function copies txt files from old directory to new, renames files \
        and creates annotation
            
    Parametres:
        old_dir(str): path of old the directory

        new_dir(str): path of new the directory

        classes(list): list filled with names of classes

        csv_name(str): name of the csv file"""
    try:
        csv_list = list()
        for c in classes:
            count = len(os.listdir(os.path.join(old_dir, c)))
            for i in range(count):
                old = os.path.abspath(os.path.join(old_dir, c, f"{i:04}.txt"))
                new = os.path.abspath(os.path.join(new_dir, f"{c}_{i:04}.txt"))
                shutil.copy(old, new)
                row = [[new, os.path.relpath(new), c]]
                csv_list += row
        create_annotation.write_into_csv(csv_name, csv_list)
    except Exception as exc:
        logging.error(f"Can not write: {exc.message}\n{exc.args}\n")


if __name__ == "__main__":
    with open(os.path.join("Lab2", "settings.json"), "r") as settings:
        settings = json.load(settings)
    copy_folder(settings["main_folder"], settings["copy"], settings["classes"], \
                f"{settings["csv"]}/{settings["copy"]}")