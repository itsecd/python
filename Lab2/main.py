"""Module providing a function printing python version 3.11.5."""
import os
import json


if __name__ == "__main__":   
    with open(os.path.join("Lab2", "json", "user_settings.json"), 'r') as settings:
        settings = json.load(settings)

    if settings["mode"] == "normal":
        import create_annotation
        create_annotation.create_csv_annotation(settings["dataset"],
                          settings["name_csv_file"],
                          settings["folder_for_csv"])
    elif settings["mode"] == "together":
        import copy_dataset_in_new_folder
        copy_dataset_in_new_folder.copy_dataset_in_new_folder(settings["new_folder_for_data"],
                                                              settings['dataset'],
                                                              settings["copy_name_csv_file"],
                                                              settings["folder_for_csv_copy"])
    elif settings["mode"] == "random":
        import copy_dataset_random_names
        copy_dataset_random_names.copy_dataset_in_new_folder(settings["random_new_folder_for_data"],
                                                             settings['dataset'],
                                                             settings["random_copy_name_csv_file"],
                                                             settings["folder_for_csv_rand"]
                                                             )
    else:
        raise Exception("invalid setting")