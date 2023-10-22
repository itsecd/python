import os


def get_path(path: str, file_name: str) -> str:
    """Returns the absolute path to the next file in the directory or None, if file doesn't exist"""
    r = os.listdir(path)
    l = r.index(file_name)
    f = os.path.join(path, r[l + 1])
    if os.path.exists(f):
        return os.path.abspath(f)
    return None
