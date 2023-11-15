import os
import logging
import shutil
import random
import json
import create_annotation


logging.basicConfig(level=logging.INFO)


def random_dataset(dataset: str, random_dataset: str, size: int, classes: list) -> list:
    """Создает папку, где файлы из random_dataset получают случайные имена."""
    random_indices = list(range(size))
    random.shuffle(random_indices)

    path_list = []
    if not os.path.exists(random_dataset):
        os.mkdir(random_dataset)

    try:
        for cls in classes:
            files_list = os.listdir(os.path.join(dataset, cls))
            for i, file_name in enumerate(files_list):
                if file_name.endswith('.txt'):
                    source_path = os.path.abspath(os.path.join(dataset, cls, file_name))
                    target_path = os.path.abspath(os.path.join(random_dataset, f'{random_indices[i]:04}.txt'))
                    shutil.copy(source_path, target_path)
                    path_set = [
                        [target_path,
                        os.path.relpath(target_path),
                        ]
                    ]
                    path_list += path_set
    except Exception as e:
        raise e  # Бросить исключение, если возникла ошибка

    return path_list


if __name__ == '__main__':
    with open(os.path.join('Lab2', 'settings.json'), 'r') as settings_file:
        settings = json.load(settings_file)

    randomized_paths = random_dataset(settings['main_dataset'],settings['dataset_random'],settings['default_number'],settings['classes'])
    create_annotation.create_annotation_file(settings['main_dataset'], settings['random_csv'])