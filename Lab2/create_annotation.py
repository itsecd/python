import csv
import os
import logging
import json


logging.basicConfig(level=logging.INFO)


def create_csv_list(dir: str, classes: list) -> list:
    """This function creates list filled with absolute path, relative path and class
                
    Parametres:
        dir(str): path of old the directory

        classes(list): list filled with names of classes
        
    Returns:
        list: list filled with csv rows"""
    csv_list = list()
    for c in classes:
        count = len(os.listdir(os.path.join(dir, c)))
        for i in range(count):
            row = [[os.path.abspath(os.path.join(dir, c, f"{i:04}.txt")), os.path.join(dir, c, f"{i:04}.txt"), c]]
            csv_list += row
    return csv_list


def write_into_csv(csv_name: str, csv_list: list) -> None:
    """This function creates csv file and writes row to the csv file
    
    Parametres: 
        csv_name(str): name of the csv file
        
        csv_list(list): list filled with csv rows"""
    try:
        for c in csv_list:
            with open(f"{csv_name}", "a") as file:
                write = csv.writer(file, lineterminator="\n")
                write.writerow(c)
    except Exception as exc: 
        logging.error(f"Can not save/write data: {exc.message}\n{exc.args}\n")


if __name__ == "__main__":
    with open(os.path.join("Lab2", "settings.json"), "r") as settings:
        settings = json.load(settings)
    csv_list = create_csv_list(settings["main_folder"], settings["classes"])
    write_into_csv(os.path.join(settings["csv"], settings["main_folder"]), csv_list)