"""Module providing a function printing python version 3.11.5."""
import pandas as pd
import os
import cv2
import logging

def filter_df_by_class(df: pd.DataFrame, class_num: int) -> pd.DataFrame:
    """
    Func filters DataFrame by class
    """
    filtered_df = df[df['label'] == class_num]
    return filtered_df

def filter_df_by_width_height(df: pd.DataFrame,
                              class_num: int,
                              max_height: int,
                              max_width:int
                              ) -> pd.DataFrame:
    filtered_df = df[df['label'] == class_num][df['width'] <= max_width]
    filtered_df = filtered_df[filtered_df['height'] <= max_height]
    return filtered_df