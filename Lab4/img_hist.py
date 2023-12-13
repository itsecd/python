import matplotlib.pyplot as plt
import numpy as np
from typing import List
        
        
def draw_hist(hists: np.ndarray) -> None:
    """The function builds a histogram based on the specified values
    10 point of lab. work"""
    colors: List[str] = ["blue", "green", "red"]
    for i in range(len(hists)):
        plt.plot(hists[i], color=colors[i], label=f"{colors[i]}")
        plt.xlim([0, 256])
    plt.ylabel("density")
    plt.title("Image Histograms")
    plt.xlabel("intensity")
    plt.legend()
    plt.show()