import os
import json
import csv_
import copy_dataset
import copy_random_num


if __name__ == "__main__":
    with open(os.path.join("Lab2", "settings.json"), "r") as f:
        settings = json.load(f)
    if settings["mode"] == "normal":
        l = csv_.make_list(settings["main_folder"], settings["tags"])
        csv_.write_scv(os.path.join(settings["directory"], settings["normal"]), l)
    if settings["mode"] == "copy":
        copy_dataset.copy_dataset(
            settings["main_folder"],
            settings["tags"],
            settings["folder"],
            (os.path.join(settings["directory"], settings["copy"])),
        )
    if settings["mode"] == "random":
        copy_random_num.copy_with_random_num(
            settings["main_folder"],
            settings["tags"],
            settings["folder"],
            (os.path.join(settings["directory"], settings["random"])),
        )
