import os

def get_next(name: str) -> str:
    """
    This function returns the relative path for the object passed to the function
    """
    path = os.path.join('dataset', name)
    img_names = os.listdir(path)
    
    for img_name in img_names:
        yield os.path.join(path, img_name)

if __name__ == "__main__":
    print(*get_next('cat'))