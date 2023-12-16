import matplotlib.pyplot as plt

def plot_histograms(hists: list):
    """
    Function plots histograms based on the provided values.
    """
    colors = ["blue", "green", "red"]
    
    for i, hist_values in enumerate(hists):
        plt.plot(hist_values, color=colors[i], label=f"Histogram {i}")
    
    plt.xlim([0, 256])
    plt.ylabel("Density")
    plt.title("Image Histograms")
    plt.xlabel("Intensity")
    plt.legend()
    plt.show()
