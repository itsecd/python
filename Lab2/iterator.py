import os
import random
import csv
import json


class ReviewIterator:
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
        for idx in range(self.current_index, len(self.file_paths)):
            if "good" in self.file_paths[idx]:
                self.current_index = idx + 1
                return self.file_paths[idx]
        return None

    def next_bad(self) -> str:
        for idx in range(self.current_index, len(self.file_paths)):
            if "bad" in self.file_paths[idx]:
                self.current_index = idx + 1
                return self.file_paths[idx]
        return None


if __name__ == "__main__":
    with open("C://Users/Ceh9/PycharmProjects/pythonProject/Lab2/options.json", "r") as options_file:
        options = json.load(options_file)
    iterator = ReviewIterator(options["output_file_annotation"], options["class_label"][0])
    for iter in iterator:
        print("Следующий экземпляр:", iter)