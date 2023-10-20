def get_path_together(object : str, number : int) -> str:
    return f"Lab1/dataset/dataset_together/{object}_{number:04}.jpg"


def get_path_random(object : str, number : int) -> str:
    return f"Lab1/dataset/dataset_random/{number:04}.jpg"


def get_path_normal(object : str, number : int) -> str:
    return f"Lab1/dataset/{object}/{number:04}.jpg"