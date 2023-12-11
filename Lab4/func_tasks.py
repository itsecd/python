import cv2
import pandas as pd
import numpy as np


def process_dataframe(df):
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


def compute_image_stats(df):
    image_size_columns = ['Height', 'Width', 'Channels']
    class_label_column = 'Label'

    image_size_stats = df[image_size_columns].describe()
    class_label_stats = df[class_label_column].value_counts()

    return image_size_stats, class_label_stats


def filter_dataframe_by_label(df, label):
    filtered_df = df[df['Label'] == label]
    return filtered_df


def filter_dataframe_by_params(df, label, max_width, max_height):
    filtered_df = df[(df['Class'] == label) & (df['Width'] <= max_width) & (df['Height'] <= max_height)]
    return filtered_df


def calculate_pixels_stats(df):
    df['Pixels'] = df['Width'] * df['Height']
    grouped_df = df.groupby('Class')['Pixels'].agg(['min', 'max', 'mean'])
    return grouped_df


def generate_histogram(df, label):
    filtered_df = df[df['Class'] == label]
    random_row = filtered_df.sample(n=1)

    image_path = random_row['Absolute path'].values[0]
    image = cv2.imread(image_path)
    b, g, r = cv2.split(image)

    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    
    return hist_b, hist_g, hist_r