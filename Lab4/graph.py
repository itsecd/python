import matplotlib.pyplot as plt
import numpy as np

def draw_histograms(histograms) -> None:
    colors = ['blue', 'green', 'red']
    channels = ['Blue Channel', 'Green Channel', 'Red Channel']

    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(np.arange(256), histograms[i], color=colors[i], label=channels[i])
        plt.title(f'{channels[i]} Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Normalized Frequency')
        plt.legend()

    plt.tight_layout()
    plt.show()