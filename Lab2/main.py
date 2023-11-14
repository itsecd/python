import os
import json
import csv_build
import randomize
import unify


if __name__ == '__main__':
    with open(os.path.join('Lab2', 'settings.json'), 'r') as settings:
        settings = json.load(settings)
    if settings['mode'] == 'path_list':
        path_list = csv_build.make_pathlist(settings['dataset_folder'], settings['classes'])
        csv_build.write_into_file(os.path.join(settings['csv_folder'], settings['pathfile_csv']), path_list)
    if settings['mode'] == 'randomize':
        path_list = randomize.randomize_dataset(os.path.join(settings['directory'], settings['dataset_folder']), 
                                      os.path.join(settings['directory'], settings['random_dataset']), 
                                      settings['classes'],  
                                      settings['default_size'])
        csv_build.write_into_file((os.path.join(settings['csv_folder'], settings['randomized_csv'])), path_list)
    if settings['mode'] == 'unify':
        path_list = unify.unify_dataset(os.path.join(settings['directory'], settings['dataset_folder']),
                                 os.path.join(settings['directory'], settings['unified_dataset']), 
                                 settings['classes'])
        csv_build.write_into_file((os.path.join(settings['csv_folder'], settings['unified_csv'])), path_list)