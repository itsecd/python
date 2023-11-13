import os
import csv
import sys

def get_next(class_label:int, csv_path='reviews.csv'):
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        read=[]
        for row in reader:
            read.append(row)
        print(len(read))
        
             


if __name__ == '__main__':
    get_next(1)