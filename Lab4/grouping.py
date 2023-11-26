"""Module providing a function printing python version 3.11.5."""
import pandas as pd
import os
import cv2
import logging
from statistics import make_describe_df

def grouping(dframe: pd.DataFrame) -> pd.DataFrame:
    """
    Func group DataFrame by label with the calculation of the maximum,
    minimum and average values by the number of pixels
    """
    dframe["pixels"] = dframe["height"] * dframe["width"]
    grouped = dframe.groupby("label")

    return grouped