import pandas as pd
import cv2
from csv_open_save import open_csv, save_csv 

def make_dframe(csv_path : str):
    dframe = open_csv(csv_path)
    abs_path = dframe["Absolute path"]
    height = []
    width = []
    channels = []
    type = []
    counter = 0
    for path in abs_path:
        img = cv2.imread(path)
        height.append(img.shape[0])
        height.append(img.shape[1])
        height.append(img.shape[2])
        if(dframe[counter, "Class"] == "rose"):
            type.append(0)
        elif(dframe[counter, "Class"] == "tulip"):
            type.append(1)

        counter+=1

    dframe["Height"] = height
    dframe["Width"] = width
    dframe["Channels"] = channels
    dframe["Type"] = type

    return dframe
