import os
import json
import csv_build


if __name__ == '__main__':
    with open(os.path.join('Lab2', 'settings.json'), 'r') as settings:
        settings = json.load(settings)
    if settings['mode'] == 'pathlist':
        pathlist = csv_build.make_pathlist(settings['dataset_folder'], settings['classes'])
        csv_build.write_into_file(os.path.join(settings['csv_folder'], settings['dataset_folder']), pathlist)