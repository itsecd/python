import csv
import os
import json

class FileIterator:
    def __init__(self, file_paths: list):
        self.file_paths = file_paths
        self.current_index = 0

    def __iter__(self):
        return self
    
    def __next__(self) -> str:
        if self.current_index < len(self.file_paths):
            current_path = self.file_paths[self.current_index]
            self.current_index += 1
            return current_path
        else:
            raise StopIteration

    def next_good(self) -> str:
        """Returns the next element for class 'good'"""
        for idx in range(self.current_index, len(self.file_paths)):
            if "good" in self.file_paths[idx]:
                self.current_index = idx + 1
                return self.file_paths[idx]
        return None

    def next_bad(self) -> str:
        """Returns the next element for class 'bad'"""
        for idx in range(self.current_index, len(self.file_paths)):
            if "bad" in self.file_paths[idx]:
                self.current_index = idx + 1
                return self.file_paths[idx]
        return None

if __name__ == "__main__":
    with open(os.path.join("Lab2", "settings.json"), "r") as settings_file:
        settings = json.load(settings_file)
    
    file_iterator = FileIterator(
        os.path.join(settings["directory"], settings["random_csv"]),
        settings["classes"][0]
    )

    for file_path in file_iterator:
        print(file_path)  # Ваша логика для обработки каждого файла