import os
import logging
import create_annotation
import shutil
import json


logging.basicConfig(level=logging.INFO)


def copy_folder(old_dir: str,new_dir: str, classes: str, csv_name:str) -> None:
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
    copy_folder(settings["main_folder"], settings["copy"], settings["classes"], f"{settings["csv"]}/{settings["copy"]}")