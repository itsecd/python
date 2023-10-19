import csv
import json
import os
import logging
import shutil

def create_csv(name_csv : str) -> None:
    try:
        if not os.path.exists(name_csv):
            with open(f"{name_csv}.csv", "w") as file:
                writer = csv.writer(file)
                writer.writerow(("Absolute path", "Relative path", "Class"))
    except Exception as ex:
        logging.error(f"Error create csv file: {ex}")

if __name__ == "__main__":
    ()