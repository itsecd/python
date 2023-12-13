import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
from pandas import DataFrame

logging.basicConfig(level=logging.INFO)


def calculate_opencv_histogram(dataframe: DataFrame, target_label: int) -> tuple:
    """
    Calculate OpenCV histograms for the RGB channels of images in the DataFrame.

    Parameters:
        dataframe: Input DataFrame.
        target_label: Numeric label for filtering.

    Returns:
        tuple: Histograms for the Blue, Green, and Red channels.
    """
    try:
        filtered_df = dataframe[dataframe['label'] == target_label]
        random_image_path = np.random.choice(
            filtered_df['absolute_path'].values)
        image = cv2.imread(random_image_path)

        b, g, r = cv2.split(image)

        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]) / b.size
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]) / g.size
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]) / r.size

        logging.info(
            f"Normalized OpenCV Histogram calculated for Class {target_label}.")
        return hist_b, hist_g, hist_r

    except Exception as e:
        logging.error(f"Error in calculate_opencv_histogram: {e}")
        return None, None, None


def plot_opencv_histogram(hist_b: np.ndarray, hist_g: np.ndarray, hist_r: np.ndarray, target_label: int) -> None:
    """
    Plot OpenCV histograms for the RGB channels using Matplotlib.

    Parameters:
        hist_b: Histogram for the Blue channel.
        hist_g: Histogram for the Green channel.
        hist_r: Histogram for the Red channel.
        target_label: Numeric label for the class.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.title(f'OpenCV Histograms for Class {target_label}')
        plt.plot(hist_b, color='blue', label='Blue Channel')
        plt.plot(hist_g, color='green', label='Green Channel')
        plt.plot(hist_r, color='red', label='Red Channel')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        logging.info(f"OpenCV Histograms plotted for Class {target_label}.")

    except Exception as e:
        logging.error(f"Error in plot_opencv_histogram: {e}")


def plot_histogram_matplotlib(hist_b: np.ndarray, hist_g: np.ndarray, hist_r: np.ndarray, target_label: int) -> None:
    """
    Plot histograms for the RGB channels using Matplotlib.

    Parameters:
        hist_b: Histogram for the Blue channel.
        hist_g: Histogram for the Green channel.
        hist_r: Histogram for the Red channel.
        target_label: Numeric label for the class.
    """
    try:
        plt.figure(figsize=(15, 8))

        plt.subplot(3, 1, 1)
        plt.plot(hist_b, color='blue')
        plt.title(f'Blue Channel Histogram for Class {target_label}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.subplot(3, 1, 2)
        plt.plot(hist_g, color='green')
        plt.title(f'Green Channel Histogram for Class {target_label}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.subplot(3, 1, 3)
        plt.plot(hist_r, color='red')
        plt.title(f'Red Channel Histogram for Class {target_label}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

        logging.info(f"Histograms plotted for Class {target_label}.")

    except Exception as e:
        logging.error(f"Error in plot_histogram_matplotlib: {e}")
