import os
import json
import generate_annotation
import randomize_dataset
import copy_dataset


if __name__ == '__main__':
    with open(os.path.join('Lab2', 'settings.json'), 'r') as settings:
        settings = json.load(settings)
    if settings['mode'] == 'normal':
        generate_annotation.generate_annotation_file(settings['dataset_folder'], settings['default_csv'])
    if settings['mode'] == 'random':
        randomize_dataset.randomize_dataset_with_annotation(settings['dataset_folder'],settings['randomized_csv'],settings['randomized_dataset'],settings['classes'],settings['default_size'])
    if settings['mode'] == 'copy':
        copy_dataset.copy_dataset_with_annotation(settings['dataset_folder'], settings['copied_dataset'], settings['copied_csv'])
            