import pandas as pd
import logging
import cv2
import matplotlib.pyplot as plt


logging.basicConfig(filename="log.log", filemode="a", level=logging.INFO)


def dataframe(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """The function generates a dataframe with columns 
    Absolute path, Height, Width, Depth and Label"""
    try:
        abs_path = df["Absolute path"]
        height_img = []
        width_img = []
        depth_img = []
        labels = []
        count = 0
        for img in abs_path:
            path = cv2.imread(img)
            tuple = path.shape
            height_img.append(tuple[0])
            width_img.append(tuple[1])
            depth_img.append(tuple[2])
            if df.loc[count, "Tag"] == tag:
                label = 0
            else:
                label = 1
            labels.append(label)
            count += 1
        df["Height"] = height_img
        df["Width"] = width_img
        df["Depth"] = depth_img
        df["Label"] = labels
        return df
    except Exception as e:
        logging.error(f"Error get img by idx{e}")


def balance(df: pd.DataFrame) -> pd.DataFrame:
    """The function receives the size of the image and 
    label and determines the balance of data"""
    img = df[["Height", "Width", "Depth"]].describe()
    label_st = df["Label"]
    label_info = label_st.value_counts()
    df = pd.DataFrame()
    tag_mentions = label_info.values
    balance = tag_mentions[0] / tag_mentions[1]
    df["Tag mentions"] = tag_mentions
    df["Balance"] = f"{balance:.1f}"
    if balance >= 0.95 and balance <= 1.05:
        logging.info("Balanced")
    else:
        logging.info(f"Not balanced,{abs(balance*100-100):.1f}%")
    return pd.concat([img, df], axis=1)


def filter_by_label(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """The function filters data by label"""
    filtered_df = df[df["Label"] == label]
    return filtered_df


def filter_with_param(
    df: pd.DataFrame, width_max: int, height_max: int, label: str
) -> pd.DataFrame:
    """The function filters data by label and maximum 
    width and height values"""
    filtered_df = df[
        (df["Label"] == label)
        & (df["Width"] <= width_max)
        & (df["Height"] <= height_max)
    ]
    return filtered_df


def groupping(df: pd.DataFrame) -> pd.DataFrame:
    """The function groups data by label, counting the 
    maximum, minimum and average number of pixels"""
    df["Pixels"] = df["Height"] * df["Width"]
    gr_df = df.groupby("Label").agg({"Pixels": ["max", "min", "mean"]})
    return gr_df


def make_histogram(df: pd.DataFrame, label: int) -> list:
    """The function creates a histogram for a random image"""
    try:
        filter_df = filter_by_label(df, label)
        img = filter_df["Absolute path"].sample().values[0]
        img_bgr = cv2.imread(img)
        height, width, channels = img_bgr.shape
        b, g, r = cv2.split(img_bgr)
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]) / (height * width)
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]) / (height * width)
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]) / (height * width)
        hists = [hist_b, hist_g, hist_r]
        return hists
    except Exception as e:
        logging.error(f"Error in make_histogram: {e}")


def draw_histogram(hists: list) -> None:
    """The function draws histograms using matplotlib"""
    colors = ["blue", "green", "red"]
    for i in range(len(hists)):
        plt.plot(hists[i], color=colors[i], label=f"histogram {i}")
        plt.xlim([0, 256])
    plt.title("Histograms")
    plt.xlabel("intensity")
    plt.ylabel("density")
    plt.legend()
    plt.show()
