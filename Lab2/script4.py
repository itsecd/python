import os


def get_path_other(path: str, file_name: str) -> str:
    """Returns the absolute path to the next file in the directory"""
    r = os.listdir(path)
    l = r.index(file_name)
    f = os.path.join(path, r[l + 1])
    return os.path.abspath(f)
