import os
import json
import create_annotation
import random_dataset
import copy_dataset


if __name__ == '__main__':
    with open(os.path.join('Lab2', 'settings.json'), 'r') as settings:
        settings = json.load(settings)
    if settings['mode'] == 'normal':
        create_annotation.create_annotation_file(settings['main_dataset'], settings['normal_csv'])
    if settings['mode'] == 'random_dataset':
        randomized_paths = random_dataset(settings['main_dataset'],settings['dataset_random'],settings['default_number'],settings['classes'])
        create_annotation.create_annotation_file(settings['main_dataset'], settings['random_csv'])
    if settings['mode'] == 'copy_dataset':
        copy_dataset(settings['main_dataset'], settings['dataset_copy'], settings['classes'])
        create_annotation.create_annotation_file((os.path.join(settings['dataset_copy'], settings['copy_csv'])))