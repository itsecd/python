import cv2
import pandas as pd
import numpy as np
from typing import Tuple, Any

def process_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.drop("Relative path", axis=1, errors='ignore')
    heights = []
    widths = []
    channels = []
    abs_paths = dataframe["Absolute path"]
    class_labels = {'brown bear': 0, 'polar bear': 1}
    dataframe['Label'] = dataframe['Class'].map(class_labels)
    for path in abs_paths:
        try:
            image = cv2.imread(path)
            if image is not None:
                height, width, channel = image.shape
                heights.append(height)
                widths.append(width)
                channels.append(channel)
            else:
                heights.append(None)
                widths.append(None)
                channels.append(None)
        except Exception as e:
            print(f"Error in image processing: {str(e)}")

    # Модификация фрейма данных
    dataframe["Height"] = heights
    dataframe["Width"] = widths
    dataframe["Channels"] = channels

    return dataframe[['Absolute path', 'Class', 'Label', 'Height', 'Width', 'Channels']]

def filter_by_label(dataframe: pd.DataFrame, class_label) -> pd.DataFrame:
    return dataframe[dataframe['Label'] == class_label]

def filter_by_parameters(dataframe: pd.DataFrame, class_label, max_height, max_width) -> pd.DataFrame:
    return dataframe[(dataframe['Class'] == class_label) &
                     (dataframe['Height'] <= max_height) &
                     (dataframe['Width'] <= max_width)]

def compute_image_stats(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    image = dataframe[["Height", "Width", "Channels"]].describe()
    label_st = dataframe["Label"]
    label_info = label_st.value_counts()
    dataframe = pd.DataFrame()
    tag_mentions = label_info.values
    balance = tag_mentions[0] / tag_mentions[1]
    dataframe["Tag mentions"] = tag_mentions
    dataframe["Balance"] = f"{balance:.1f}"
    if balance >= 0.95 and balance <= 1.05:
        print("Balanced")
    return pd.concat([image, dataframe], axis=1)


def group_by_stats(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['Pixels'] = dataframe.apply(lambda row: row['Height'] * row['Width'], axis=1)
    grouped_stats = dataframe.groupby('Class')['Pixels'].agg(['max', 'min', 'mean'])
    return grouped_stats



