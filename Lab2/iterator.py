import csv
import os
import json


class PathIterator:
    """This Iterator returns path of the class. When class ends, raises\
        StopIteration"""
    def __init__(self, csv_path: str, name_class: str):
        self.data = list()
        self.count = 0
        self.mark = name_class
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if self.mark == row[2]:
                    self.data.append(row[0])

    def __iter__(self):
        return self
    def __next__(self) -> str:
        if self.count < len(self.data):
            self.count += 1
            return self.data[self.count-1]
        else:
            return None

class ClassIterator:
    def __init__(self, path, first_class, second_class, third_class, fourth_class, fifth_class):
        self.__first_iterator = PathIterator(path, first_class)
        self.__second_iterator = PathIterator(path, second_class)
        self.__third_iterator = PathIterator(path, third_class)
        self.__fourth_iterator = PathIterator(path, fourth_class)
        self.__fifth_iterator = PathIterator(path, fifth_class)

    def next_first(self):
        return next(self.__first_iterator)
    
    def next_second(self):
        return next(self.__second_iterator)
    
    def next_third(self):
        return next(self.__third_iterator)
    
    def next_fourth(self):
        return next(self.__fourth_iterator)
    
    def next_fifth(self):
        return next(self.__fifth_iterator)


if __name__ == "__main__":
    with open(os.path.join("Lab2", "settings.json"), "r") as settings:
        settings = json.load(settings)
    iter = ClassIterator(os.path.join(settings["csv"], settings["normal_csv"]), settings["classes"][0],settings["classes"][1],settings["classes"][2],settings["classes"][3],settings["classes"][4])
    for i in range(500):
        print(iter.next_first())