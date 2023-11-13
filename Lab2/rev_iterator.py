import os
import csv
import argparse

class RevIterator:
    def __init__(self,path_to_csv:str,class_label:int) -> None:
        self.class_label=class_label
        self.path_to_csv=path_to_csv
        self.reviews=[]
        self.counter=0

        with open(path_to_csv, newline='') as csvfile:
            for row in csv.reader(csvfile, delimiter=','):
                if (class_label==int(row[-1])):
                    self.reviews.append(row)


    def __iter__(self):
        return self
    
    def __next__(self) -> str:
        if self.counter < len(self.reviews):
            self.counter += 1
            return self.reviews[self.counter-1][0]
        else:
            raise StopIteration
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input csv path, label of class")
    parser.add_argument("-c", "--csv", help="Input csv path", type=str)
    parser.add_argument("-l", "--label", help="Input label of class", type=int)
    args = parser.parse_args()
    iter = RevIterator(args.csv,args.label)
    for i in iter:
        print(i)