import matplotlib.pyplot as plt
import numpy as np


def plot_histograms(hist_blue: np.ndarray, hist_green: np.ndarray, hist_red: np.ndarray) -> None:
    '''a function for drawing histograms'''
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title('Blue Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(hist_blue, color='blue')

    plt.subplot(1, 3, 2)
    plt.title('Green Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(hist_green, color='green')
  
    plt.subplot(1, 3, 3)
    plt.title('Red Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(hist_red, color='red')
    
    plt.tight_layout()
    plt.show()