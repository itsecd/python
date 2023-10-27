import os
import json
import csv_annotation
import new_name_copy
import random_of_copy

if __name__ == "__main__":

    with open(os.path.join("Lab2", "settings.json"), "r") as settings:
        settings = json.load(settings)


if settings["mode"] == "normal":
    l = csv_annotation.make_list(settings["main_folder"], settings["classes"])
    csv_annotation.write_in_file(
        (os.path.join(settings["directory"],
         settings["folder"], settings["normal"])), l
    )

if settings["mode"] == "new_name":
    new_name_copy.copy_in_new_directory(
        settings["main_folder"],
        settings["classes"],
        settings["main_folder"],
        (os.path.join(settings["directory"],
         settings["folder"], settings["new_name"])),
    )

if settings["mode"] == "random":
    random_of_copy.copy_with_random(
        settings["main_folder"],
        settings["classes"],
        settings["main_folder"],
        (os.path.join(settings["directory"],
         settings["folder"], settings["random"])),
    )
