import matplotlib.pyplot as plt


def draw_histogram(hists:list):
    """The function builds a histogram based on the specified values
    10 point of lab. work"""
    colors = ["blue", "green", "red"]
    for i in range(len(hists)):
        plt.plot(hists[i], color=colors[i], label=f"Histogram {i}")
        plt.xlim([0, 256])
    plt.ylabel("density")
    plt.title("Image Histograms")
    plt.xlabel("intensity")
    plt.legend()
    plt.show()