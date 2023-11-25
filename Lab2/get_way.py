import os

def get_path_normal(directory : str, object : str, name : str) -> str:
    """
    Get path in normal format

    Parametrs
    ---------
    object : str
        Name of object
    name : str
        Name of image
    """
    return os.path.join(directory, object, name)


def get_path_together(save_fold : str, object : str, name : str) -> str:
    """
    Get path in together format

    Parametrs
    ---------
    object : str
        Name of object
    name : int
        Name of image
    """
    return os.path.join(save_fold, f"{object}_{name}")


def get_path_random(save_fold : str, number : str) -> str:
    """
    Get path in random format

    Parametrs
    ---------
    object : str
        Name of object
    number : int
        Number of object
    """
    return os.path.join(save_fold, f"{number:04}.jpg")
