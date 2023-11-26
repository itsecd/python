"""Module providing a function printing python version 3.11.5."""
import pandas as pd
import cv2
import os
import logging
import json
import numpy as np
from statistics import make_describe_df
from grouping import grouping
from filter import filter_df_by_label, filter_df_by_width_height_label
from build_histogram import build_histogram, show_histogram


logging.basicConfig(level=logging.DEBUG)


def set_width_height_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Adds columns with width height and depth of the image in DataFrame"""
    for i in range(df.shape[0]):
        try:
            path = df['absolute_path'].iloc[i]
            image = cv2.imread(path)

            width, height, depth = image.shape
            
            df.loc[i, 'height'] = height
            df.loc[i, 'width'] = width
            df.loc[i, 'depth'] = depth
            
        except Exception as e:
            logging.exception(e)
    df.fillna(0, inplace=True)
    df[['height', 'width', 'depth']] = df[['height', 'width', 'depth']].astype (int)
    return df


def is_balanced(df: pd.DataFrame) -> bool:
    """Checking for the balance of the DataFrame by class"""
    class_stats = df['class'].value_counts()
    return class_stats.min() / class_stats.max() >= 0.8


def main_func(path_annotation:str,
              max_width: int,
              max_height: int,
              label: int) -> None:
    """
    Main func that combines the remaining functions
    for consistent execution according to the terms of reference
    """
    df = pd.read_csv(path_annotation)
    df = df[['absolute_path', 'class']]
    df.columns = ['absolute_path', 'class']
    df['label'] = df['class'].map({'tiger': 0, 'leopard': 1})
    set_width_height_depth(df)
    make_describe_df(df)
    grouping(df)
    print(df)
    max_pixel_count = df['pixels'].max()
    min_pixel_count = df['pixels'].min()
    mean_pixel_count = df['pixels'].mean()
    show_histogram(df, label)
    logging.info(max_pixel_count, "\n", min_pixel_count, "\n", mean_pixel_count, "\n")
    #print("balanced: ", is_balanced(df))
    #print(filter_df_by_width_height_label(df, label, max_width, max_height))


if __name__ == "__main__":
    with open(os.path.join("Lab4","json","user_settings.json"), "r") as f:
        settings = json.load(f)
    main_func(settings['path_annotation'],
              settings['max_width'],
              settings['max_height'],
              settings['label']
              )