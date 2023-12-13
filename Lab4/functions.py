# functions.py
import cv2
import numpy as np
import pandas as pd

def add_pixel_count_column(df):
    df['количество_пикселей'] = df.apply(lambda row: get_pixel_count(row['абсолютный_путь_к_файлу']), axis=1)
    return df

def get_pixel_count(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        return np.prod(image.shape)
    else:
        return np.nan

def group_by_class_and_compute_stats(df):
    grouped_df = df.groupby('название_класса')['количество_пикселей'].agg(['min', 'max', 'mean'])
    return grouped_df
