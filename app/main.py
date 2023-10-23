import os
import json
import make_normal, make_together, make_random


if __name__ == "__main__":   
    path_to_dir = os.path.dirname(__file__)

    with open(os.path.join(path_to_dir, "json", "src.json"), 'r') as src_json:
        src = json.load(src_json)

    with open(os.path.join(path_to_dir, "json", "setting.json"), 'r') as setting_json:
        setting = json.load(setting_json)

    with open(os.path.join(path_to_dir, "json", "name.json"), 'r') as name_json:
        name = json.load(name_json)

    if setting["mode"] == "normal":
        make_normal.MakeNormalCsv.write_in_file(os.path.join(src["name-dir"], src["name-csv-dir"], name["name-normal"]), setting["objects"], src["name-images-dir"])
    elif setting["mode"] == "together":
        if not os.path.isdir(os.path.join(src["name-dir"], src["name-data-dir"], name["name-together"])):
            os.mkdir(os.path.join(src["name-dir"], src["name-data-dir"], name["name-together"]))

        make_together.MakeTogetherData.make_new_fold(os.path.join(src["name-dir"], src["name-csv-dir"], name["name-together"]), setting["objects"], src["name-images-dir"])
    elif setting["mode"] == "random":
        if not os.path.isdir(os.path.join(src["name-dir"], src["name-data-dir"], name["name-random"])):
            os.mkdir(os.path.join(src["name-dir"], src["name-data-dir"], name["name-random"]))

        make_random.MakeRandomData.make_new_fold(os.path.join(src["name-dir"], src["name-csv-dir"], name["name-random"]), setting["objects"], src["name-images-dir"])
    else:
        raise Exception("invalid setting")