import csv
import logging

logging.basicConfig(filename="py_log1.log", filemode="a", level=logging.INFO)

def get_path(name_csv:str, tag: str) -> None:
    """Function returns the next instance
     of the tag """
    logging.info("get_path")
    l=[]
    with open(name_csv,"r") as f:
        reader=csv.reader(f)
        for row in reader:
            l.append(row)
    limit=len(l)
    for i in range(limit):
        if l[i][2]== tag:
            count=i-1
            break
    if count<limit:
        if tag == l[count+1][2]:
            count+=1
        return l[count][0]
    else:
        return None


if __name__=="__main__":
    logging.info(get_path("file.csv","tiger"))
