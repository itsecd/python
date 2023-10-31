import os
import json
from make_csv import make_csv

if __name__ == "__main__":   
    path_to_dir = os.path.dirname(__file__)
    with open(os.path.join(path_to_dir, "json", "setting.json"), 'r') as setting_json:
        setting = json.load(setting_json)

    path_to_normal = os.path.join(setting["name-dir"], 
                              setting["name-csv-dir"], 
                              setting["name-normal"])
    
    path_to_together = os.path.join(setting["name-dir"], 
                                    setting["name-data-dir"], 
                                    setting["name-together"])


    path_to_random = os.path.join(setting["name-dir"], 
                                    setting["name-data-dir"], 
                                    setting["name-random"])
    if setting["mode"] == "normal":
        make_csv(path_to_normal, 
                 setting["objects"], 
                 setting["name-images-dir"],
                 setting["mode"])
    elif setting["mode"] == "together":
        if not os.path.isdir(path_to_together):
            os.mkdir(path_to_together)

        make_csv(path_to_together, 
                 setting["objects"], 
                 setting["name-images-dir"],
                 setting["mode"])
    elif setting["mode"] == "random":
        if not os.path.isdir(path_to_random):
            os.mkdir(path_to_random)

        make_csv(path_to_random, 
                 setting["objects"], 
                 setting["name-images-dir"],
                 setting["mode"])
    else:
        raise Exception("invalid setting")