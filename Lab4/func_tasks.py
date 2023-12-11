import cv2
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Any


logging.basicConfig(level=logging.INFO)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    '''a function for reading csv file data and creating a new csv with other columns'''
    try:
        df = df.drop("Relative path", axis=1)
        
        heights = []
        widths = []
        channels = []
        abs_paths = df["Absolute path"]
        counter=0
        class_labels = {'polar bear': 0, 'brown bear': 1}
        df['Label'] = df['Class'].map(class_labels)
        
        for path in abs_paths:
            img = cv2.imread(path)
            if img is not None:
                height, width, channel = img.shape
                heights.append(height)
                widths.append(width)
                channels.append(channel)
                print(counter)
                counter+=1
            else:
                heights.append(None)
                widths.append(None)
                channels.append(None)
        
        df['Height'] = heights
        df['Width'] = widths
        df['Channels'] = channels
        
        return df[['Absolute path', 'Class', 'Label', 'Height', 'Width', 'Channels']]
    except:
        logging.error(f"Recording error")


def compute_image_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    '''a function to determine the balance of the collected dataframe'''
    try:
        image_size_columns = ['Height', 'Width', 'Channels']
        class_label_column = 'Label'

        image_size_stats = df[image_size_columns].describe()
        class_label_stats = df[class_label_column].value_counts()

        return image_size_stats, class_label_stats
    except:
        logging.error(f"Balance check error")

def filter_dataframe_by_label(df: pd.DataFrame, label: Any) -> pd.DataFrame:
    '''function for filtering data frame by label'''
    try:
        filtered_df = df[df['Label'] == label]
        return filtered_df
    except:
        logging.error(f"Filter for label error")


def filter_dataframe_by_params(df: pd.DataFrame, label: str, max_width: int, max_height: int) -> pd.DataFrame:
    '''function for filtering data frame by width and height'''
    try:
        filtered_df = df[(df['Class'] == label) & (df['Width'] <= max_width) & (df['Height'] <= max_height)]
        return filtered_df
    except:
        logging.error(f"Filter for params error")

def calculate_pixels_stats(df: pd.DataFrame) -> pd.DataFrame:
    '''a function for grouping a DataFrame by class label with calculation of the maximum, minimum and average values by the number of pixels'''
    df['Pixels'] = df['Width'] * df['Height']
    grouped_df = df.groupby('Class')['Pixels'].agg(['min', 'max', 'mean'])
    return grouped_df


def generate_histogram(df: pd.DataFrame, label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''a function for creating histograms based on data obtained from a dataset'''
    try:
        filtered_df = df[df['Class'] == label]
        random_row = filtered_df.sample(n=1)

        image_path = random_row['Absolute path'].values[0]
        image = cv2.imread(image_path)
        b, g, r = cv2.split(image)

        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
        
        return hist_b, hist_g, hist_r
    except Exception as ex:
        logging.error(f"File for histogram was not found: {ex.message}\n{ex.args}\n")