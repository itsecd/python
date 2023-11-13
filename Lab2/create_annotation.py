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


