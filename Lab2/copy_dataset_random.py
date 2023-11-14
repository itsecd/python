import os
import shutil
import csv
import random

def replace_images_and_randomize(class_name: str, 
                                 source_path: str, 
                                 dest_path: str) -> None:
    """
    This function renames images by combining the image number, class name, and random number in the format class_name_number.jpg,
    transfers the images to the destination directory, and deletes the folder where the class images were stored.
    """
    class_path = os.path.join(source_path, class_name)
    image_names = os.listdir(class_path)
    image_rel_paths = [os.path.join(class_path, name) for name in image_names]
    
    new_names = [f'{class_name}_{i}.jpg' for i in range(len(image_names))]
    new_rel_paths = [os.path.join(dest_path, name) for name in new_names]

    for old_name, new_name in zip(image_rel_paths, new_rel_paths):
        os.replace(old_name, new_name)

    os.chdir(source_path)

    if os.path.isdir(class_name):
        shutil.rmtree(class_name)

    os.chdir('..')

def main() -> None:
    if os.path.isdir('dataset1'):
        shutil.rmtree('dataset1')

    old_path = os.path.relpath('dataset')
    new_path = os.path.relpath('dataset1')

    shutil.copytree(old_path, new_path)

    replace_images_and_randomize('brown bear', new_path, new_path)
    replace_images_and_randomize('polar bear', new_path, new_path)

    if os.path.isdir('dataset2'):
        shutil.rmtree('dataset2')

    old_path = os.path.relpath('dataset1')
    new_path = os.path.relpath('dataset2')

    shutil.copytree(old_path, new_path)

    old_names = os.listdir(new_path)
    old_rel_paths = [os.path.join(new_path, name) for name in old_names]

    random_numbers = random.sample(range(0, 10001), len(old_names))
    new_names = [f'{name.split("_")[0]}_{random_number}.jpg' for name, random_number in zip(old_names, random_numbers)]
    new_rel_paths = [os.path.join(new_path, name) for name in new_names]

    for old_name, new_name in zip(old_rel_paths, new_rel_paths):
        os.replace(old_name, new_name)

    new_full_paths = [os.path.join(os.path.abspath(new_path), name) for name in new_names]

    with open('paths2.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=';', lineterminator='\r')
        for full_path, rel_path, old_rel_path in zip(new_full_paths, new_rel_paths, old_rel_paths):
            class_name = old_rel_path.split("_")[0].replace("dataset2\\", "")
            writer.writerow([full_path, rel_path, class_name])
    shutil.rmtree('dataset1')

if __name__ == "__main__":
    main()
