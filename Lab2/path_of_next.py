import csv
import argparse
from annotation import write_csv, AnnotationLabel

def get_path_of_next(class_label: int, ind: int, csv_path='reviews.csv'):
    """
    Return the path of the elem of the class "class_label"
    next after the element whose index is passed.
    """
    with open(csv_path, newline='') as csvfile:
        files = [row for row in csv.reader(csvfile, delimiter=',') if class_label == int(row[-1])]
        return None if ind >= len(files) else files[ind][0]  
             

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input csv path, label of class, index, label of annotation")
    parser.add_argument("-c", "--csv", help="Input csv path", type=str)
    parser.add_argument("-l", "--label", help="Input label of class", type=int)
    parser.add_argument("-i", "--index", help="Input index", type=int)
    parser.add_argument("-a", "--annotation", help="Input label of annotation", type=int)
    parser.add_argument("-n", "--new", help="Input path to new dir", type=str)
    parser.add_argument("-o", "--old", help="Input path to old dir", type=str)
    args = parser.parse_args()
    #-c review.csv -l 1 -i 3 -a 2 -n new_dataset -o dataset
    print(get_path_of_next(args.label,args.index,(write_csv(args.csv,args.annotation,args.new,args.old))))
    