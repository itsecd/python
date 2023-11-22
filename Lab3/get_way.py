import os

def get_path_normal(directory : str, object : str, number : int) -> str:
    """
    Get path in normal format

    Parametrs
    ---------
    object : str
        Name of object
    number : int
        Number of object
    """
    return os.path.join(directory, object, f"{number:04}.jpg")


def get_path_together(save_fold : str, object : str, number : int) -> str:
    """
    Get path in together format

    Parametrs
    ---------
    object : str
        Name of object
    number : int
        Number of object
    """
    return os.path.join(save_fold, f"{object}_{number:04}.jpg")


def get_path_random(save_fold : str, number : int) -> str:
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
