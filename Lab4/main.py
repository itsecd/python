import pandas as pd
from csv_open_save import open_csv, save_csv 

def make_dframe(csv_path : str):
    dframe = open_csv(csv_path)
    abs_path = dframe["Absolute path"]
    type = []
    counter = 0
    for img in abs_path:
        if(dframe[counter, "Class"] == "rose"):
            type.append(0)
        elif(dframe[counter, "Class"] == "tulip"):
            type.append(1)

        counter+=1

    dframe["Type"] = type

    return dframe
