import numpy as np
import matplotlib.pyplot as plt


def draw_histograms(hist_blue: np.ndarray, hist_green: np.ndarray, hist_red: np.ndarray) -> None:
    '''A function for drawing histograms'''
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    colors = ['blue', 'green', 'red']
    channels = ['Blue', 'Green', 'Red']

    for ax, hist, color, channel in zip(axs, [hist_blue, hist_green, hist_red], colors, channels):
        ax.set_title(f'{channel} Channel Histogram')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.plot(hist, color=color)

    plt.tight_layout()
    plt.show()
