"""Module providing a function printing python version 3.11.5."""
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def build_histogram(df: pd.DataFrame, label: int) -> list:
    """build_histogram of random image in DataFrame"""
    filtered_df = df[df['label'] == label]
    random_img_idx = np.random.randint(0, len(filtered_df))
    img_path = filtered_df.iloc[random_img_idx, 0]

    img = cv2.imread(img_path)
    channels = cv2.split(img)

    width = df.iloc[random_img_idx]["width"]
    height = df.iloc[random_img_idx]["height"]
    depth = df.iloc[random_img_idx]["depth"]
    answer = []
    for i in range(depth):
        hist = cv2.calcHist([channels[i]], [0], None, [256], [0, 256]).flatten() / (width * height)
        answer.append(hist)

    return answer


def show_histogram(hists: list) -> None:
    """
    draw histograms that are returned from the function of paragraph 9.
    Graphs and axes must have appropriate signatures.
    """
    blue, green, red = hists
    plt.hist(blue, alpha=0.5,  color='blue', label='blue')
    plt.hist(green, alpha=0.5, color='green', label='green')
    plt.hist(red, alpha=0.5, color='red', label='red')

    plt.xlabel('Saturation')
    plt.ylabel('Density')
    plt.title('Histogram of picture')

    plt.legend()
    plt.show()
