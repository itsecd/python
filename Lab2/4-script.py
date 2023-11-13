import csv


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
    print(get_path_of_next(1,1))