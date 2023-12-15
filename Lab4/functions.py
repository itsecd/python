import cv2
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Any


logging.basicConfig(level=logging.INFO)


def process_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.drop("Relative Path", axis=1, errors='ignore')
    heights = []
    widths = []
    channels = []
    abs_paths = dataframe["Absolute Path"]
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
            logging.error(f"Error in image processing")

    dataframe["Height"] = heights
    dataframe["Width"] = widths
    dataframe["Channels"] = channels

    return dataframe[['Absolute Path', 'Class', 'Label', 'Height', 'Width', 'Channels']]


def filter_by_label(dataframe: pd.DataFrame, class_label) -> pd.DataFrame:
    '''This function filter dataframe by label'''
    return dataframe[dataframe['Label'] == class_label]


def filter_by_parameters(dataframe: pd.DataFrame, class_label, max_height, max_width) -> pd.DataFrame:
    '''This function filter dataframe by parameters'''
    return dataframe[(dataframe['Class'] == class_label) &
                     (dataframe['Height'] <= max_height) &
                     (dataframe['Width'] <= max_width)]


def extract_image_stats(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    '''This function extract stats of image'''
    label_column = 'Label'
    size_columns = ['Height', 'Width', 'Channels']
    label_stats = dataframe[label_column].value_counts()
    size_stats = dataframe[size_columns].describe()
    if len(label_stats) > 1:
        logging.info("Dataset is balanced")
    else:
        logging.info("Dataset may be unbalanced")
    return size_stats, label_stats


def group_by_stats(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''This function groups a DataFrame by class label with calculation of the maximum, minimum and average values by the number of pixels'''
    dataframe['Pixels'] = dataframe['Width'] * dataframe['Height']
    grouped_stats = dataframe.groupby('Class')['Pixels'].agg(['max', 'min', 'mean'])
    return grouped_stats


def create_histogram(dataframe: pd.DataFrame, class_label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''This function create histograms'''
    filter_df = dataframe[dataframe['Class'] == class_label]
    random_row = filter_df.sample(n=1)
    image_path = random_row['Absolute Path'].values[0]
    image = cv2.imread(image_path)
    height,width=image.shape
    b, g, r = cv2.split(image)

    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])/(height*width)
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])/(height*width)
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])/(height*width)
    
    return hist_b, hist_g, hist_r