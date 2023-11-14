import os


def get_next(class_name: str) -> str:
    """
    This function returns the relative path for the class object passed
    to the function
    """
    path = os.path.join('dataset', class_name)
    class_names = os.listdir(path)
    class_names.append(None)
    for i in range(len(class_names)):
        if class_names[i] is not None:
            yield os.path.join(path, class_names[i])
        elif class_names[i] is None:
            yield None   


print(*get_next('polar bear'))
print(*get_next('brown bear'))