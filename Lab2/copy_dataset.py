import os
import logging
import shutil
import json
import create_annotation

logging.basicConfig(level=logging.INFO)


def copy_dataset(dataset: str, copy_dataset: str, classes: list) -> None:
    """Копирует файлы из dataset в copy_dataset с переименованием по формату {class}_{number}.txt."""
    path_list = []
    if not os.path.exists(copy_dataset):
        os.mkdir(copy_dataset)
    try:
        for cls in classes:
            files_list = os.listdir(os.path.join(dataset, cls))
            for i, file_name in enumerate(files_list):
                if file_name.endswith('.txt'):
                    source_path = os.path.abspath(os.path.join(dataset, cls, file_name))
                    target_path = os.path.abspath(os.path.join(copy_dataset, f'{cls}_{i+1:04}.txt'))
                    shutil.copy(source_path, target_path)
                    path_set = [
                        [target_path,
                         os.path.relpath(target_path, copy_dataset),
                         cls]
                    ]
                    path_list += path_set

        logging.info(f"Файлы из {dataset} успешно скопированы в {copy_dataset}")
    except Exception as e:
        logging.error(f"Произошла ошибка в copy_dataset: {e}", exc_info=True)



if __name__ == '__main__':
    with open(os.path.join("Lab2", "settings.json"), "r") as settings_file:
        settings = json.load(settings_file)

    copy_dataset(settings['main_dataset'], settings['dataset_copy'], settings['classes'])
    create_annotation.create_annotation_file(settings['main_dataset'], settings['copy_csv'])