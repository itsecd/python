def get_path_together(object : str, number : int) -> str:
    return f"dataset/{object}_{number:04}.jpg"


def get_path_random(object : str, number : int) -> str:
    return f"dataset/{number:04}.jpg"


def get_path_normal(object : str, number : int) -> str:
    return f"dataset/{object}/{number:04}.jpg"