import pandas as pd
import matplotlib.pyplot as plt
import cv2
from csv_open_save import open_csv

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
    if coefficient >= 0.98 and coefficient <= 1.02:
        print("DataFrame is Balanced")
    else:
        print(f"DataFrame is not Balanced\nCoefficient: {coefficient}")

    stats_frame = dframe[["Height", "Width", "Channels"]].describe()
    return pd.DataFrame.join(type_count, stats_frame)
    

def filter_by_type(dframe : pd.DataFrame, type : int) -> pd.DataFrame:
    return pd.DataFrame(dframe["Type"] == type)

def filter_by_size(dframe : pd.DataFrame,
                   type : int,
                   max_height : int,
                   max_width : int) -> pd.DataFrame:
    return pd.DataFrame(filter_by_type(dframe, type) and dframe["Height"] <= max_height
                        and dframe["Width"] <= max_width)


def grouping(dframe : pd.DataFrame) -> pd.DataFrame:
    dframe["Pixels"] = dframe["Height"] * dframe["Width"]
    dframe.groupby("Type").agg({"Pixels": ["max", "min", "mean"]})


def make_hists(dframe : pd.DataFrame, type : int) -> list:
    dframe_type = filter_by_type(type)
    img = cv2.imread(dframe_type["Absolute path"].sample().values[0])
    height, width, channels = img.shape
    b, g, r = cv2.split(img)
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])/(height*width)
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])/(height*width)
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])/(height*width)
    hists = [hist_b, hist_g, hist_r]
    return hists

def draw_hists(hists :  list) -> None:
    plt.plot(hists[0], color='blue', label='Blue') 
    plt.plot(hists[1], color='green', label='Green')
    plt.plot(hists[2], color='blue', label='Red')

    plt.xlabel("intensity")
    plt.ylabel("density")
    plt.title("Histograms")
    plt.legend()
    plt.show()

    