import os

def get_next(name: str) -> str:
    """
    This function returns the relative path for the object passed to the function
    """
    path = os.path.join('dataset', name)
    img_names = os.listdir(path)
    img_names.append(None)
    
    for i in range(len(img_names)):
        if img_names[i] is not None:
            yield os.path.join(path, img_names[i])
        elif img_names[i] is None:
            yield None   

if __name__ == "__main__":
    print(*get_next('cat'))