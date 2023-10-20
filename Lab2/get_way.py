def get_path_together(object : str, number : int) -> str:
    """
    Get path in together format

    Get path in together format
    Parametrs
    ---------
    object : str
        Name of object
    number : int
        Number of object
    """
    return f"Lab1/dataset/dataset_together/{object}_{number:04}.jpg"


def get_path_random(object : str, number : int) -> str:
    """
    Get path in random format

    Get path in random format
    Parametrs
    ---------
    object : str
        Name of object
    number : int
        Number of object
    """
    return f"Lab1/dataset/dataset_random/{number:04}.jpg"


def get_path_normal(object : str, number : int) -> str:
    """
    Get path in normal format

    Get path in normal format
    Parametrs
    ---------
    object : str
        Name of object
    number : int
        Number of object
    """
    return f"Lab1/dataset/{object}/{number:04}.jpg"