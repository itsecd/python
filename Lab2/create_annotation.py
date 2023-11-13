import csv
import os
import json
import logging


logging.basicConfig(level=logging.INFO)


def create_csv(name: str) -> None:
    try: 
        if not os.path.exists(name):
            with open(f"{name}.csv", "a") as file:
                csv.writer(file, lineterminator="\n")
    except Exception as exc:
        logging.error(f"Can not create file: {exc.message}\n{exc.args}\n")


def create_csv_list(directory: str, classes: str) -> list:
    csv_list = list()
    for star in classes:
        count = len(os.listdir(os.path.join(directory, star)))
        for i in range(count):
            r = [[os.path.abspath(os.path.join(directory, star, f"{i:04}.txt")), os.path.join(directory, star, f"{i:04}.txt"), star]]
            csv_list += r
    return csv_list

