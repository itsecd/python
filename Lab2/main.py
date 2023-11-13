import os
import json
import create_annotation
import create_copy_folder
import create_copy_random


if __name__ == "__main__":
    with open(os.path.join("Lab2", "settings.json"), "r") as settings:
        settings = json.load(settings)
    if settings["mode"] == "normal":
        l = create_annotation.create_csv_list(settings["main_folder"], settings["classes"])
        create_annotation.write_into_csv(f"{settings["csv"]}/{settings["main_folder"]}", l)
    if settings["mode"] == "copy":
        create_copy_folder.copy_folder(settings["main_folder"], settings["copy"], settings["classes"], f"{settings["csv"]}/{settings["copy"]}")
    if settings["mode"] == "random":
        create_copy_random.copy_random(settings["main_folder"], settings["random"], settings["classes"], f"{settings["csv"]}/{settings["random"]}")