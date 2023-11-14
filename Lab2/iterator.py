import csv
import os
import json
class SimpleIterator:
    def __init__(self, eggs, choice, search):
        self.ch = []
        with open(f'{eggs}.csv', 'r') as r_file:
            file_reader = csv.reader(r_file, delimiter = ",")
            for a in file_reader:
                self.ch.append(a)
            self.max = len(self.ch)
            if not os.path.exists(eggs):
                with open(f'{choice}.csv', 'w') as csvfile: 
                    spamwriter = csv.writer(csvfile, lineterminator="\n")
                    spamwriter.writerow(["Path"])
                    for i in range(self.max):
                        if self.ch[i + 1][2] == search:
                            self.count = i - 1
                            break
        self.search = search
        self.choice = choice
        self.eggs = eggs
        self.count += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.max:
            if not os.path.exists(self.eggs):
                with open(f'{self.choice}.csv', 'a') as csvfile:  
                    spamwriter = csv.writer(csvfile, lineterminator="\n")
                    if self.ch[self.count + 1][2] == self.search:
                        spamwriter.writerow({self.ch[self.count + 1][1]})
                    self.count += 1
        else:
            return None

class ChoiceIterator:
    def __init__(self, eggs, path, cat, dog):
        self.__first_iterator = SimpleIterator(eggs, path, cat)
        self.__second_iterator = SimpleIterator(eggs, path, dog)

    def next_cat(self):
        return next(self.__first_iterator)

    def next_dog(self):
        return next(self.__second_iterator)

if __name__ == "__main__":
    with open(os.path.join("Lab1", "main.json"), "r") as main_file:
        main = json.load(main_file)

    iterator = ChoiceIterator(main["folder_an"], main["folder_iterator"], "cat", "dog")
    cat_path = iterator.next_cat()
    cat_path1 = iterator.next_cat()
    cat_path2 = iterator.next_cat()
    dog_path = iterator.next_dog()
    dog_path1 = iterator.next_dog()
