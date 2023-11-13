import csv
import argparse
from annotation import write_csv 


def get_path_of_next(class_label:int,ind:int,csv_path='reviews.csv'):
    """
    the function returns the path of the element next after the element
    whose index is passed.
    class_label : int
    ind : int
    csv_path : str
    """
    with open(csv_path, newline='') as csvfile:
        files=[]
        for row in csv.reader(csvfile, delimiter=','):
            if (class_label==int(row[-1])):
                files.append(row)
        cnt=0
        for file in files:
            if cnt!=ind:
                cnt+=1
            else:
                return file[0] 
        return None
        
             

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
    