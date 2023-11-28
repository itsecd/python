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
    def __init__(self, path, cls0, cls1, cls2, cls3, cls4, cls5):
        self.__iter0 = PathIterator(path, cls0)
        self.__iter1 = PathIterator(path, cls1)
        self.__iter2 = PathIterator(path, cls2)
        self.__iter3 = PathIterator(path, cls3)
        self.__iter4 = PathIterator(path, cls4)
        self.__iter5 = PathIterator(path, cls5)

    def next_0(self):
        return next(self.__iter0)

    def next_1(self):
        return next(self.__iter1)
    
    def next_2(self):
        return next(self.__iter2)
    
    def next_3(self):
        return next(self.__iter3)
    
    def next_4(self):
        return next(self.__iter4)
    
    def next_5(self):
        return next(self.__iter5)


if __name__ == "__main__":
    with open(os.path.join("Lab2", "settings.json"), "r") as settings:
        settings = json.load(settings)
    iter = ClassIterator(os.path.join(settings["csv_folder"], settings["pathfile_csv"]), settings["classes"][0],settings["classes"][1],settings["classes"][2],settings["classes"][3],settings["classes"][4],settings["classes"][5])
    for i in range(5):
        print(iter.next_1())