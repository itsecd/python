"""Module providing a function printing python version 3.11.5."""
import os
import json
import logging
import csv


logging.basicConfig(level=logging.DEBUG)

def object_by_index(name_csv: str, index: int, object: str) -> str:
    """Using the csv library and annotation, the function returns the absolute path to the desired element"""
    with open(name_csv, "r", newline='') as f:
        reader = csv.DictReader(f)
        class_iterator = 0
        for row in reader:
            if row['class'] == object:
                if(class_iterator == index):
                    return row['absolute_path']
                class_iterator += 1
    raise IndexError("Element not found")

                

if __name__ == "__main__":
    with open(os.path.join("Lab2","user_settings.json"), "r", newline='') as f:
        settings = json.load(f)
    logging.info(object_by_index(settings["annotation_for_iterating"],
                                 settings["number_of_iteration"],
                                 settings["class"]
                                 ))