import os
from typing import Generator

def get_next(name: str) -> Generator[str, None, None]:
    """
    This function yields the relative path for each object in the specified directory.
    """
    path: str = os.path.join('dataset', name)
    img_names = os.listdir(path)  # Не указываем тип явно

    for img_name in img_names:
        yield os.path.join(path, img_name)


if __name__ == "__main__":
    cat_generator = get_next('cat')

    next_cat_image = next(cat_generator)
    print(next_cat_image)

    next_cat_image = next(cat_generator)
    print(next_cat_image)
