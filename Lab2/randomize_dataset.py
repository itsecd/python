import os
import random
import csv

def copy_with_random_filename(source_path: str, destination_folder: str) -> str:
    random_filename = f"{random.randint(0, 10000)}.jpg"
    destination_path_random = os.path.join(destination_folder, random_filename)

    with open(source_path, 'rb') as source_file, open(destination_path_random, 'wb') as destination_file:
        destination_file.write(source_file.read())

    return destination_path_random

def randomize_dataset(dataset_path: str, destination_path: str) -> None:
    with open('annotation.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            _, relative_path, _ = row
            source_path = os.path.join(dataset_path, relative_path)
            copy_with_random_filename(source_path, destination_path)

if __name__ == "__main__":
    randomize_dataset('dataset', 'randomized_dataset')
