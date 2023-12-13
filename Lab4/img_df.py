import pandas as pd
import cv2
import numpy as np
import logging
from typing import List


def df_from_csv(path: str) -> pd.DataFrame | None:
    """This function for reading data from 'csv' and converting it into a DataFrame\n
       with the addition of new attribute columns: 'width'; 'height'; 'class'; 'channels'
    """
    try:
        df: pd.DataFrame = pd.read_csv(path, delimiter=",")
        keys = list(df.columns.values)
        del df[keys[1]]    
        df['class'] = pd.factorize(df[keys[2]])[0]
        imgs_data = np.array(list(df.apply(get_img_props, axis=1))).transpose()
        df["width"] = imgs_data[1]
        df["height"] = imgs_data[0]    
        df["channels"] = imgs_data[2]
        return df
    except FileNotFoundError as ex:
        logging.error(ex) 


def get_img_props(row: pd.Series) -> tuple[int] | None:
    """This function get image properties as like: width, height, channels count"""
    keys = list(row.to_dict())
    try:
        img: cv2.MatLike = cv2.imread(row[keys[0]])
        return img.shape[0], img.shape[1], img.shape[2]
    except FileNotFoundError as ex:
        logging.error(ex)    


def df_imgs_stats(df: pd.DataFrame, ratio_acc: float = 0.05) -> pd.DataFrame:
    """This function calc statistic-data for attrs as like: width, height, channels, class"""
    images_info: pd.DataFrame = df[["height", "width", "channels"]].describe()       
    df_stats: pd.DataFrame = class_ratio_check(df, ratio_acc)    
    return pd.concat([images_info, df_stats], axis=1)


def class_ratio_check(df: pd.DataFrame, accuracy: float = 0.05) -> pd.DataFrame:
    """This function calc the ratio of classes in a sample"""
    label_info: pd.DataFrame = df['class'].value_counts()
    tag_mentions: pd.ndarray = label_info.values
    perf_ratio: float = 1/len(tag_mentions)
    ratio: List[float] = [] 
    df_stats = pd.DataFrame()
    df_stats["Quantity"] = tag_mentions
    is_balanced: bool = True
    for tag in tag_mentions:
        ratio.append(tag/len(df))
        if not abs(ratio[-1] - perf_ratio) <= accuracy:
            is_balanced = False 
    df_stats["Balance"] = ratio  
    if is_balanced:
        logging.info("df is balanced")
    else:
        logging.info("df is disalanced")   
    return df_stats            


def filter_by_class_lbl(df: pd.DataFrame, class_lbl: int = None) -> pd.DataFrame:
    """This function returns a filtered selection by class label"""
    if class_lbl:
        df: pd.DataFrame = df[df["class"] == class_lbl]    
    return df


def filter_by_imgsize(df: pd.DataFrame, wmax: int = None, hmax: int = None) -> pd.DataFrame:
    """This function returns a filtered selection by size-image charactheristics"""
    if wmax:
        df: pd.DataFrame = df[df["width"] <= wmax]
    if hmax:
        df: pd.DataFrame = df[df["height"] <= hmax]        
    return df


def filter_by(df: pd.DataFrame, wmax: int = None, hmax: int = None, class_lbl: int = None) -> pd.DataFrame:
    """This function returns a filtered selection by class label"""
    return filter_by_imgsize(filter_by_class_lbl(df, class_lbl), wmax, hmax)


def group_by_resolution(df: pd.DataFrame) -> pd.DataFrame:
    """This function returns a filtered sample based on the specified characteristics"""
    df["resolution"] = df["height"] * df["width"]    
    return df.groupby("class").agg({"resolution": ["max", "min", "mean"]})


def build_hist(df: pd.DataFrame, class_lbl: int) -> np.ndarray | None:
    """this function returns statistical data\n
       about the distribution of colors across channels in the form of a histogram
    """
    df: pd.Dataframe = filter_by_class_lbl(df, class_lbl)
    img_path: str = df["absolute path"].sample().values[0]
    try:
        image: cv2.Matlike = cv2.imread(img_path)
        b, g, r = cv2.split(image)
        res: float = image.shape[0]*image.shape[1]
        hist_b: cv2.Matlike = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g: cv2.Matlike = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r: cv2.Matlike = cv2.calcHist([r], [0], None, [256], [0, 256])    
        return np.array([hist_b/res, hist_g/res, hist_r/res])
    except FileNotFoundError as ex:
        logging.error(ex)      
