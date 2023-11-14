import os
import json
import csv_
import copy_dataset


if __name__ == "__main__":
    with open(os.path.join("Lab2", "settings.json"), "r") as f:
        settings = json.load(f)
    if settings["mode"] == "normal":
        l = csv_.make_list(settings["main_folder"], settings["tags"])
        csv_.write_csv(os.path.join(settings["directory"], settings["normal"]), l)
    if settings["mode"] == "copy":
        copy_dataset.copy_dataset(
            settings["main_folder"],
            settings["tags"],
            settings["folder"],
            (os.path.join(settings["directory"], settings["copy"])),
            settings["bool"]
        )
    if settings["mode"] == "random":
        copy_dataset.copy_dataset(
            settings["main_folder"],
            settings["tags"],
            settings["folder"],
            (os.path.join(settings["directory"], settings["random"])),
            settings['bool']
        )
