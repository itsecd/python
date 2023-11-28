import csv
import os
import json


class ElementIterator:
    def __init__(self, name_csv, choice, search):
        self.ch = []
        if not os.path.exists(name_csv):
            with open(f'{name_csv}.csv', 'r') as r_file:
                file_reader = csv.reader(r_file, delimiter=",")
                for a in file_reader:
                    self.ch.append(a)
                self.max = len(self.ch)
                for i in range(self.max):
                    if self.ch[i][2] == search:
                        self.count = i - 1
                        break
        self.__search = search
        self.__choice = choice
        self.__name_csv = name_csv

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.max:
            if not os.path.exists(self.__name_csv):
                with open(f'{self.__choice}.csv', 'a') as csvfile:
                    spamwriter = csv.writer(csvfile, lineterminator="\n")
                    if self.ch[self.count + 1][2] == self.__search:
                        spamwriter.writerow({self.ch[self.count + 1][1]})
                    self.count += 1
                    return self.ch[self.count][1]
        else:
            return None


class ChoiceIterator:
    def __init__(self, name_csv, path, cat, dog):
        self.__first_iterator = ElementIterator(name_csv, path, cat)
        self.__second_iterator = ElementIterator(name_csv, path, dog)

    def next_cat(self):
        return next(self.__first_iterator)

    def next_dog(self):
        return next(self.__second_iterator)


if __name__ == "__main__":
    with open(os.path.join("Lab2", "main.json"), "r") as main_file:
        main = json.load(main_file)

    iterator = ChoiceIterator(
        main["folder_an"], main["folder_iterator"], "cat", "dog")
    cat_path = iterator.next_cat()
    cat_path1 = iterator.next_cat()
    cat_path2 = iterator.next_cat()
    dog_path = iterator.next_dog()
    dog_path1 = iterator.next_dog()
