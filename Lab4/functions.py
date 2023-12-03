import pandas as pd
import matplotlib.pyplot as plt
import cv2
import logging

logging.basicConfig(level=logging.INFO)

def make_dframe(dframe: pd.DataFrame) -> pd.DataFrame:
    """
    Function add Height, Width and Type of image in dataframe
    """
    try:
        abs_path = dframe["Absolute path"]
        height = []
        width = []
        channels = []
        type = []
        counter = 0
        for path in abs_path:
            img = cv2.imread(path)
            cv_tuple = img.shape
            height.append(cv_tuple[0])
            width.append(cv_tuple[1])
            channels.append(cv_tuple[2])
            if dframe.loc[counter, "Class"] == "rose":
                label = 0
            else:
                label = 1
            type.append(label)
            counter += 1

        dframe["Height"] = height
        dframe["Width"] = width
        dframe["Channels"] = channels
        dframe["Type"] = type

        return dframe
    except Exception as ex:
        logging.error(f"Make frame error: {ex}")


def make_stats(dframe : pd.DataFrame) -> pd.DataFrame:
    """
    Make dframe that consists discribe and check balanced about input dataframe
    """
    try:
        type = dframe["Type"]
        type_count = type.value_counts().values
        coefficient = type_count[0]/type_count[1]
        df = pd.DataFrame()
        df["Quantity"] = type_count
        df["Balance"] = f"{coefficient:.1f}"
        if coefficient >= 0.98 and coefficient <= 1.02:
            logging.info("DataFrame is Balanced")
        else:
            logging.info(f"DataFrame is not Balanced\nCoefficient: {coefficient}")

        stats_frame = dframe[["Height", "Width", "Channels"]].describe()
        return pd.concat([stats_frame, df], axis=1)
    except Exception as ex:
        logging.error(f"Balance check error: {ex}")
    

def filter_by_type(dframe : pd.DataFrame, class_ : str) -> pd.DataFrame:
    """
    Filter dataframe by type of image
    """
    return dframe[dframe["Class"] == class_]


def filter_by_size(dframe : pd.DataFrame,
                   type : int,
                   max_height : int,
                   max_width : int) -> pd.DataFrame:
    """
    Filter dataframe by max height, width and type of image
    """
    return pd.DataFrame(filter_by_type(dframe, type) and dframe["Height"] <= max_height
                        and dframe["Width"] <= max_width)


def grouping(dframe : pd.DataFrame) -> pd.DataFrame:
    """
    Grouping dframe by Type
    """
    dframe["Pixels"] = dframe["Height"] * dframe["Width"]
    return dframe.groupby("Type").agg({"Pixels": ["max", "min", "mean"]})
    

def make_hists(dframe : pd.DataFrame, type : int) -> list:
    """
    Make histogramm by dataframe
    """
    try:
        dframe_type = filter_by_type(dframe, type)
        img = cv2.imread(dframe_type["Absolute path"].sample().values[0])
        height, width, channels = img.shape
        b, g, r = cv2.split(img)
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])/(height*width)
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])/(height*width)
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])/(height*width)
        hists = [hist_b, hist_g, hist_r]
        return hists
    except Exception as ex:
        logging.error(f"Make historam error: {ex}")


def draw_hists(hists :  list) -> None:
    """
    Draw Histogramm
    """
    plt.plot(hists[0], color='blue', label='Blue') 
    plt.plot(hists[1], color='green', label='Green')
    plt.plot(hists[2], color='blue', label='Red')

    plt.xlabel("intensity")
    plt.ylabel("density")
    plt.title("Histograms")
    plt.legend()
    plt.show()  