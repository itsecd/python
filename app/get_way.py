import json, os

def json_unpack():
    path_to_dir = os.path.dirname(__file__)

    with open(os.path.join(path_to_dir, "json", "src.json"), 'r') as src_json:
        src = json.load(src_json)

    with open(os.path.join(path_to_dir, "json", "setting.json"), 'r') as setting_json:
        setting = json.load(setting_json)

    with open(os.path.join(path_to_dir, "json", "name.json"), 'r') as name_json:
        name = json.load(name_json)

    return [os.path.join(src["name-images-dir"]), os.path.join(src["name-dir"], src["name-data-dir"], name["name-together"]), os.path.join(src["name-dir"], src["name-data-dir"], name["name-random"])]


def get_path_normal(object : str, number : int) -> str:
    """
    Get path in normal format

    Parametrs
    ---------
    object : str
        Name of object
    number : int
        Number of object
    """
    return os.path.join(json_unpack()[0], f"{object}/{number:04}.jpg")


def get_path_together(object : str, number : int) -> str:
    """
    Get path in together format

    Parametrs
    ---------
    object : str
        Name of object
    number : int
        Number of object
    """
    return os.path.join(json_unpack()[1], f"{object}_{number:04}.jpg")


def get_path_random(object : str, number : int) -> str:
    """
    Get path in random format

    Parametrs
    ---------
    object : str
        Name of object
    number : int
        Number of object
    """
    return os.path.join(json_unpack()[2], f"{number:04}.jpg")
