import pandas as pd
import cv2
from csv_open_save import open_csv, save_csv 

def make_dframe(csv_path : str) -> pd.DataFrame:
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


def make_stats(dframe : pd.DataFrame) -> pd.DataFrame:
    type_count = dframe["Type"].value_counts().values
    coefficient = type_count[0]/type_count[1]
    if coefficient >= 0.95 and coefficient <= 1.05:
        print("DataFrame is Balanced")
    else:
        print(f"DataFrame is not Balanced\nCoefficient: {coefficient}")

    stats_frame = dframe[["Height", "Width", "Channels"]].describe()
    return pd.DataFrame.join(type_count, dframe.describe())
    

def filter_by_type(dframe : pd.DataFrame, type : int) -> pd.DataFrame:
    return pd.DataFrame(dframe["Type"] == type)

def filter_by_size(dframe : pd.DataFrame,
                   type : int,
                   max_height : int,
                   max_width : int) -> pd.DataFrame:
    return pd.DataFrame(filter_by_type(dframe, type) and dframe["Height"] <= max_height
                        and dframe["Width"] <= max_width)


