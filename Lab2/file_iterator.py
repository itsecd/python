import csv
import os
import json


class FileIterator:
    def __init__(self, csv_path: str, class_name: str):
        self.file_paths = []
        self.class_name = class_name
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if self.class_name == row[2]:
                    self.file_paths.append(row[0])

    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self) -> str:
        if self.current_index < len(self.file_paths):
            current_path = self.file_paths[self.current_index]
            self.current_index += 1
            return current_path
        else:
            raise StopIteration


if __name__ == "__main__":
    with open(os.path.join("Lab2", "settings.json"), "r") as settings_file:
        settings = json.load(settings_file)
    
    file_iterator = FileIterator(
        os.path.join(settings["directory"], settings["random_csv"]),
        settings["classes"][0]
    )

    for file_path in file_iterator:
        print(file_path)