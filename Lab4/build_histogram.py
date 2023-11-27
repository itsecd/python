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

    hist_blue = np.zeros((256,))
    hist_green = np.zeros((256,))
    hist_red = np.zeros((256,))

    for channel in channels:
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
        if channel is channels[0]:
            hist_blue = hist
        elif channel is channels[1]:
            hist_green = hist
        else:
            hist_red = hist

    return hist_blue, hist_green, hist_red


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
