"""Module providing a function printing python version 3.11.5."""
import pandas as pd
import os
import cv2
import logging


logging.basicConfig(level=logging.DEBUG)


def make_describe_df(df: pd.DataFrame,
                     name_file:str = "statistic.csv",
                     path:str = os.path.join("Lab4","csv_files")
                     ) -> pd.DataFrame:
    """This function creates a csv file describing the DataFrame"""
    try:
        if not os.path.exists(path):
                os.makedirs(path)
        stats = df.describe()
        stats.to_csv(os.path.join(path, name_file))
    except Exception as e:
         logging.exception(e)
    return stats
