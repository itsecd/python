import pandas as pd
import cv2

def add_image_parameters(dframe: pd.DataFrame, class_one: str) -> pd.DataFrame:
    """
    Function reads a CSV file as a DataFrame and adds new columns:
    image parameters (height, width, and number of channels),
    and assigns a unique class label (0 or 1).
    """
    try:
        heights, widths, channels, labels = [], [], [], []
        for path_of_image, class_label in zip(dframe["Absolute path"], dframe["Class"]):
            img = cv2.imread(path_of_image)
            height, width, channel = img.shape
            heights.append(height)
            widths.append(width)
            channels.append(channel)
            label = 0 if class_label == class_one else 1
            labels.append(label)

        dframe["Height"] = heights
        dframe["Width"] = widths
        dframe["Channels"] = channels
        dframe["Label"] = labels
        return dframe
    except Exception as ex:
        pass


def check_balance(dframe: pd.DataFrame) -> pd.DataFrame:
    """
    Function accepts a CSV file, calculates statistics, and checks the balance of the dataset.
    """
    try:
        images_info = dframe[["Height", "Width", "Channels"]].describe()
        label_info = dframe["Label"].value_counts()
        df = pd.DataFrame({"Quantity": label_info.values})
        is_balanced = label_info.iloc[0] / label_info.iloc[1]
        df["Balance"] = f"{is_balanced:.1f}"

        return pd.concat([images_info, df], axis=1)
    except Exception as ex:
        pass


def filter_by_label(dframe: pd.DataFrame, label: int) -> pd.DataFrame:
    """Function returns a DataFrame filtered by the class label."""
    return dframe[dframe["Label"] == label]


def filter_by_min_max(
    dframe: pd.DataFrame, width_max: int, height_max: int, label: int
) -> pd.DataFrame:
    """Function returns a DataFrame filtered by the class label and set min-max values."""
    return dframe[
        (dframe["Label"] == label)
        & (dframe["Width"] <= width_max)
        & (dframe["Height"] <= height_max)
    ]


def group_by_label(dframe: pd.DataFrame) -> pd.DataFrame:
    """Function groups the DataFrame by the class label with calculation of max, min, and mean values by pixels."""
    dframe["Pixels"] = dframe["Height"] * dframe["Width"]
    grouped = dframe.groupby("Label").agg({"Pixels": ["max", "min", "mean"]})
    return grouped


def build_histogram(dframe: pd.DataFrame, label: int) -> list:
    """Function builds a histogram for a random image and returns a list of arrays for each channel."""
    try:
        filtered_df = filter_by_label(dframe, label)
        image_path = filtered_df["Absolute path"].sample().values[0]
        image_bgr = cv2.imread(image_path)
        height, width, _ = image_bgr.shape
        b, g, r = cv2.split(image_bgr)
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]) / (height * width)
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]) / (height * width)
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]) / (height * width)
        hists = [hist_b, hist_g, hist_r]
        return hists
    except Exception as ex:
        pass
