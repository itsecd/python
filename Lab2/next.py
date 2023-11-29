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


    '''
                                spamwriter.writerow({self.ch[i + 1][1]})
                            elif self.ch[i + 1 + main['max_file']][2] == search:
                                spamwriter.writerow({self.ch[i + 1 + main['max_file']][1]})
                            else:
                                return None
        self.ch = []
        self.count = -1
        with open(f'{eggs}.csv', 'r') as r_file:
            file_reader = csv.reader(r_file, delimiter = ",")
            for a in file_reader:
                self.ch.append(a)


            if not os.path.exists(eggs):
                with open(f'{choice}.csv', 'w') as csvfile: 
                    spamwriter = csv.writer(csvfile, lineterminator="\n")
                    spamwriter.writerow(["Path"])


            for i in range(main["max_file"]):
                print(self.ch[i + 1][2] , search)
                if self.ch[i + 1][2] == search:
                    self.count = i - 1
                    break'''
        

    '''
            if not os.path.exists(eggs):
                with open(f'{choice}.csv', 'w') as csvfile: 
                    spamwriter = csv.writer(csvfile, lineterminator="\n")
                    spamwriter.writerow(["Path"])
                    for i in range(main["max_file"]):
                        print(self.ch[i+1][2], search)
                        if self.ch[i + 1][2] == search:
                            spamwriter.writerow({self.ch[i + 1][1]})
                        else:
                            spamwriter.writerow({self.ch[main["max_file"] + i + 1][1]})'''
                            

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.max:
            print(self.count, self.max)
            if not os.path.exists(self.eggs):
                with open(f'{self.choice}.csv', 'a') as csvfile:  
                    spamwriter = csv.writer(csvfile, lineterminator="\n")
                    if self.ch[self.count + 1][2] == self.search:
                        spamwriter.writerow({self.ch[self.count + 1][1]})
                        print(self.ch[self.count + 1][1], self.search)
                    #else:
                        #spamwriter.writerow({self.ch[self.count + main['max_file'] + 1][1]})
                        #print(self.ch[self.count + main['max_file'] + 1][1], self.search)
                    
                    self.count += 1
                    print(self.count)
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
    cat_path3 = iterator.next_cat()
    cat_path4 = iterator.next_cat()
    cat_path5 = iterator.next_cat()
    cat_path6 = iterator.next_cat()
    cat_path7 = iterator.next_cat()
    cat_path8 = iterator.next_cat()
    cat_path9 = iterator.next_cat()
    cat_path10= iterator.next_cat()
    cat_path11 = iterator.next_cat()
    cat_path12 = iterator.next_cat()
    cat_path13 = iterator.next_cat()
    cat_path14 = iterator.next_cat()
    cat_path15 = iterator.next_cat()
    cat_path16 = iterator.next_cat()
    cat_path17 = iterator.next_cat()
    cat_pat18h = iterator.next_cat()
    cat_path19 = iterator.next_cat()
    cat_path20 = iterator.next_cat()
    cat_path21 = iterator.next_cat()
    cat_path22 = iterator.next_cat()
    cat_path23 = iterator.next_cat()
    cat_path24 = iterator.next_cat()
    cat_path25 = iterator.next_cat()
    cat_path26 = iterator.next_cat()
    cat_path27 = iterator.next_cat()
    cat_path28 = iterator.next_cat()
    cat_path29 = iterator.next_cat()
    cat_path30 = iterator.next_cat()
    cat_path31 = iterator.next_cat()
    cat_path32 = iterator.next_cat()
    cat_path33 = iterator.next_cat()
    cat_path34 = iterator.next_cat()
    cat_path35 = iterator.next_cat()
    cat_path36 = iterator.next_cat()
    dog_path = iterator.next_dog()
    dog_path1 = iterator.next_dog()
    dog_path2 = iterator.next_dog()
    dog_path3 = iterator.next_dog()
    dog_path4 = iterator.next_dog()
    dog_path5 = iterator.next_dog()
    dog_path6 = iterator.next_dog()
    dog_path7 = iterator.next_dog()
    dog_path8 = iterator.next_dog()
    dog_path9 = iterator.next_dog()

'''
class ElementIterator:
    """The iterator class gets the path to the cvs file and the class,
    returns the next element if it is of the same class, else it returns None"""

    def __init__(self, name_csv: str, name_class: str) -> None:
        """The method initializes the initial values of the iterator"""
        self.data = list()
        self.count = -1
        with open(name_csv, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                self.data.append(row)
                self.count = 0
        self.limit = len(self.data)
        for i in range(self.limit):
            if self.data[i + 1][2] == name_class:
                self.count = i - 1
                break
        self.mark = name_class
        self.name_csv = name_csv

    def __iter__(self):
        """Returns the iterator itself"""
        return self

    def __next__(self) -> str:
        """Returns the absolute path to the next class element"""
        if self.count < self.limit:
            if self.mark == self.data[self.count + 1][2]:
                self.count += 1
            return self.data[self.count][0]
        else:
            return None


class ClassesIterator:

    def __init__(self, path, first_class, second_class):
        self.__first_iterator = ElementIterator(path, first_class)
        self.__second_iterator = ElementIterator(path, second_class)

    def next_first(self):
        return next(self.__first_iterator)

    def next_second(self):
        return next(self.__second_iterator)


if __name__ == "__main__":

    iter = ClassesIterator("Lab2\csv_files\datasets.csv", "rose", "tulip")
    next_rose_path = iter.next_first()
    next_rose_path1 = iter.next_first()'''




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


    '''
                                spamwriter.writerow({self.ch[i + 1][1]})
                            elif self.ch[i + 1 + main['max_file']][2] == search:
                                spamwriter.writerow({self.ch[i + 1 + main['max_file']][1]})
                            else:
                                return None
        self.ch = []
        self.count = -1
        with open(f'{eggs}.csv', 'r') as r_file:
            file_reader = csv.reader(r_file, delimiter = ",")
            for a in file_reader:
                self.ch.append(a)


            if not os.path.exists(eggs):
                with open(f'{choice}.csv', 'w') as csvfile: 
                    spamwriter = csv.writer(csvfile, lineterminator="\n")
                    spamwriter.writerow(["Path"])


            for i in range(main["max_file"]):
                print(self.ch[i + 1][2] , search)
                if self.ch[i + 1][2] == search:
                    self.count = i - 1
                    break'''
        

    '''
            if not os.path.exists(eggs):
                with open(f'{choice}.csv', 'w') as csvfile: 
                    spamwriter = csv.writer(csvfile, lineterminator="\n")
                    spamwriter.writerow(["Path"])
                    for i in range(main["max_file"]):
                        print(self.ch[i+1][2], search)
                        if self.ch[i + 1][2] == search:
                            spamwriter.writerow({self.ch[i + 1][1]})
                        else:
                            spamwriter.writerow({self.ch[main["max_file"] + i + 1][1]})'''
                            

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.max:
            #print(self.count, self.max)
            if not os.path.exists(self.eggs):
                with open(f'{self.choice}.csv', 'a') as csvfile:  
                    spamwriter = csv.writer(csvfile, lineterminator="\n")
                    if self.ch[self.count + 1][2] == self.search:
                        spamwriter.writerow({self.ch[self.count + 1][1]})
                        print(self.ch[self.count + 1][1], self.search)
                    self.count += 1
                    print(self.count)
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
    cat_path3 = iterator.next_cat()
    cat_path4 = iterator.next_cat()
    cat_path5 = iterator.next_cat()
    dog_path = iterator.next_dog()
    dog_path1 = iterator.next_dog()
    dog_path2 = iterator.next_dog()
    dog_path3 = iterator.next_dog()




'''
class ElementIterator:
    """The iterator class gets the path to the cvs file and the class,
    returns the next element if it is of the same class, else it returns None"""

    def __init__(self, name_csv: str, name_class: str) -> None:
        """The method initializes the initial values of the iterator"""
        self.data = list()
        self.count = -1
        with open(name_csv, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                self.data.append(row)
                self.count = 0
        self.limit = len(self.data)
        for i in range(self.limit):
            if self.data[i + 1][2] == name_class:
                self.count = i - 1
                break
        self.mark = name_class
        self.name_csv = name_csv

    def __iter__(self):
        """Returns the iterator itself"""
        return self

    def __next__(self) -> str:
        """Returns the absolute path to the next class element"""
        if self.count < self.limit:
            if self.mark == self.data[self.count + 1][2]:
                self.count += 1
            return self.data[self.count][0]
        else:
            return None


class ClassesIterator:

    def __init__(self, path, first_class, second_class):
        self.__first_iterator = ElementIterator(path, first_class)
        self.__second_iterator = ElementIterator(path, second_class)

    def next_first(self):
        return next(self.__first_iterator)

    def next_second(self):
        return next(self.__second_iterator)


if __name__ == "__main__":

    iter = ClassesIterator("Lab2\csv_files\datasets.csv", "rose", "tulip")
    next_rose_path = iter.next_first()
    next_rose_path1 = iter.next_first()


///////////////////
    try:
        img_list = []
        for tag in tags:
            if choice == 1:
                random_num = set()
                count_files = len(os.listdir(os.path.join(old_dir, tags[0])))
                while len(random_num) <= (count_files * len(tags)):
                    random_num.add(random.randint(0, 1000))
                numbers = list(random_num)
                for i in range(count_files):
                    j = len(os.listdir(os.path.join(new_dir))) - len(tags)
                    b = os.path.abspath(os.path.join(old_dir, tag, f"{i:04}.jpg"))
                    a = os.path.abspath(os.path.join(new_dir, f"{numbers[j]:04}.jpg"))
                    shutil.copy(b, a)
                    a = [[new, os.path.relpath(new), s]]
                    string += a
            else:
                count_files = len(os.listdir(os.path.join(old_dir, tag)))
                for i in range(count_files):
                    b = os.path.abspath(os.path.join(old_dir, tag, f"{i:04}.jpg"))
                    a = os.path.abspath(os.path.join(new_dir, f"{tag}_{i:04}.jpg"))
                    shutil.copy(b, a)
                    img = [[a, os.path.realpath(a), tag]]
                    img_list += img

/////////
import os
import json
import logging
import new

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    with open(os.path.join("Lab2", "main.json"), "r") as main_file:
        main = json.load(main_file)

    new.write_in_new(main["folder"], main["search"], main["folder_random"], 1)'''