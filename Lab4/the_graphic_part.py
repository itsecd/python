import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging


logging.basicConfig(level=logging.INFO)


def calculate_opencv_histogram(dataframe, target_label):
    try:
        filtered_df = dataframe[dataframe['label'] == target_label]
        random_image_path = np.random.choice(
            filtered_df['absolute_path'].values)
        image = cv2.imread(random_image_path)

        b, g, r = cv2.split(image)

        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

        logging.info(f"OpenCV Histogram calculated for Class {target_label}.")
        return hist_b, hist_g, hist_r

    except Exception as e:
        logging.error(f"Error in calculate_opencv_histogram: {e}")
        return None, None, None


def plot_opencv_histogram(hist_b, hist_g, hist_r, target_label):
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


def plot_histogram_matplotlib(hist_b, hist_g, hist_r, target_label):
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
