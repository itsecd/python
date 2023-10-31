import json, os

def json_unpack() -> list:
    """
    Open jsons files
    """

    with open(os.path.join(os.path.dirname(__file__), "json", "setting.json"), 'r') as setting_json:
        setting = json.load(setting_json)

    return [setting["name-images-dir"], 
            os.path.join(setting["name-dir"], setting["name-data-dir"], setting["name-together"]), 
            os.path.join(setting["name-dir"], setting["name-data-dir"], setting["name-random"])]


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
    return os.path.join(json_unpack()[0], object, f"{number:04}.jpg")


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
