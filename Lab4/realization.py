import cv2
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)


def generate_frame(frame: pd.DataFrame, label_class: str) -> pd.DataFrame:
    '''сформировать DataFrame из названия класса и абсолютного пути к файлу; 
    добавление столбца с меткой и трех столбцов, содержащих информацию об изображении'''
    try:
        height_image = []
        width_image = []
        depth_image = []
        number_label = []
        count = 0
        for path in frame["Absolute path"]:
            image = cv2.imread(path)
            cv_image = image.shape
            height_image.append(cv_image[0])
            width_image.append(cv_image[1])
            depth_image.append(cv_image[2])
            if frame.loc[count, "Class"] == label_class:
                label = 0
            else:
                label = 1
            number_label.append(label)
            count += 1
        frame["Height"] = height_image
        frame["Width"] = width_image
        frame["Depth"] = depth_image
        frame["Label"] = number_label
        return frame
    except:
        logging.error("Error in generate_frame \n")


def balance_test(frame: pd.DataFrame) -> pd.DataFrame:
    '''вычислить статистическую информацию для столбцов; определить, является ли набор сбалансированным'''
    try:
        label_info = frame["Label"].value_counts()
        balanced = label_info.values[0] / label_info.values[1]
        if balanced >= 0.9 and balanced <= 1.1:
            logging.info(f"Набор сбалансирован \n")
        else:
            logging.info("Набор не сбалансирован \n")
        return frame[["Height", "Width", "Depth"]].describe()
    except:
        logging.error("Errot in balance_test")


def filter(frame: pd.DataFrame, label: int) -> pd.DataFrame:
    '''возвращает отфильтрованный по метке DataFrame'''
    filter_frame = frame[frame["Label"] == label] 
    return filter_frame


def max_filter(frame: pd.DataFrame, width_max: int, height_max: int, label: int) -> pd.DataFrame:
    '''фильтрация по максимальному значению'''
    filter_frame = frame[(frame["Height"] <= height_max) & (
        frame["Width"] <= width_max) & (frame["Label"] == label)] 
    return filter_frame


def grouping(frame: pd.DataFrame) -> pd.DataFrame:
    '''группировка по количеству пикселей'''
    frame["Pixels"] = frame["Height"] * frame["Width"]
    frame_groupy = frame.groupby(["Label"]).agg(
        {"Pixels": ["max", "min", "mean"]}) 
    return frame_groupy
