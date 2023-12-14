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

def image_stats(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    label_column = 'Label'
    size_columns = ['Height', 'Width', 'Channels']
    label_stats = dataframe[label_column].value_counts()
    size_stats = dataframe[size_columns].describe()
    if len(label_stats) > 1:
        print("Dataset is balanced")
    else:
        print("Dataset may be unbalanced")
    return size_stats, label_stats


def group_by_stats(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['Pixels'] = dataframe['Width'] * dataframe['Height']
    grouped_stats = dataframe.groupby('Class')['Pixels'].agg(['max', 'min', 'mean'])
    return grouped_stats

def create_histogram(dataframe: pd.DataFrame, class_label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    filter_df = filter_by_label(dataframe, class_label)
    random_row = filter_df.sample(n=1)
    image_path = random_row['Absolute path'].values[0]
    image = cv2.imread(image_path)
    height,width=image.shape
    b, g, r = cv2.split(image)

    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])/(height*width)
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])/(height*width)
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])/(height*width)
    
    return hist_b, hist_g, hist_r