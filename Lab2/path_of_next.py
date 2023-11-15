import os

def get_next(name: str) -> str:
    """
    This function returns the relative path for the next object in the specified directory.
    """
    path = os.path.join('dataset', name)
    img_names = os.listdir(path)

    for img_name in img_names:
        yield os.path.join(path, img_name)

if __name__ == "__main__":
    cat_generator = get_next('cat')

    next_cat_image = next(cat_generator)
    print(next_cat_image)

    next_cat_image = next(cat_generator)
    print(next_cat_image)
