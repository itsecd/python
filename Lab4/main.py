import pandas as pd
import logging
import cv2
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


def image_forms(csv_path: str, new_csv_filename: str, class_one) -> None:
    """Function reads the csv-file as a dateframe and adds new columns to it:
    image parameters (height, width and channels) and also set a unique class label
    1-4 points of lab. work"""
    try:
        height_image = []
        width_image = []
        channels_image = []
        numerical = []
        counter = 0
        dframe = pd.read_csv(
            csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"]
        )
        abs_path = dframe["Absolute path"]
        for path_of_image in abs_path:
            img = cv2.imread(path_of_image)
            cv_tuple = img.shape
            height_image.append(cv_tuple[0])
            width_image.append(cv_tuple[1])
            channels_image.append(cv_tuple[2])
            if dframe.loc[counter, "Class"] == class_one:
                label = 0
            else:
                label = 1
            numerical.append(label)
            counter += 1
        dframe = dframe.drop("Relative path", axis=1)
        dframe["Height"] = height_image
        dframe["Width"] = width_image
        dframe["Channels"] = channels_image
        dframe["Label"] = numerical
        dframe.to_csv(new_csv_filename, index=False)
    except:
        logging.error(f"Recording error: {ex.message}\n{ex.args}\n")


def checking_balance(dataframe_file: str, statistic_file: str) -> None:
    """The function accepts a csv file, calculates statistics
    and also checks the dataset for balance
    5 point of lab. work"""
    try:
        dframe = pd.read_csv(dataframe_file)
        images_info = dframe[["Height", "Width", "Channels"]].describe()
        label_stats = dframe["Label"]
        label_info = label_stats.value_counts()
        df = pd.DataFrame()
        num_images_per_label = label_info.values
        is_balanced = num_images_per_label[0] / num_images_per_label[1]
        df["Quantity"] = num_images_per_label
        df["Balance"] = f"{is_balanced:.1f}"
        pd.concat([images_info, df], axis=1).to_csv(statistic_file)
        if is_balanced >= 0.95 and is_balanced <= 1.05:
            logging.info(f"Выборка сбалансированна")
        else:
            logging.info(
                "Выборка несбалансирована, погрешность:",
                f"{abs(is_balanced*100-100):.1f}%",
            )
    except:
        logging.error(f"Balance check error")


def filter_by_label(dframe, label) -> pd.DataFrame:
    """The function returns a dataframe filtered by the class label
    6 point of lab. work"""
    filtered_df = dframe[dframe["Label"] == label]
    return filtered_df


def min_max_filter(dframe, width_max: int, height_max: int, label: str) -> pd.DataFrame:
    """The function returns a dataframe filtered
    by the class label and set min-max values
    7 point of lab. work"""
    filtered_df = dframe[
        (dframe["Label"] == label)
        & (dframe["Width"] <= width_max)
        & (dframe["Height"] <= height_max)
    ]
    return filtered_df


def grouping(dframe) -> pd.DataFrame:
    """Function groups the DataFrame by the class label
    with the calculation of the max, min and mean values by the number of pixels
    8 point of lab. work"""
    dframe["Pixels"] = dframe["Height"] * dframe["Width"]
    grouped = dframe.groupby("Label").agg({"Pixels": ["max", "min", "mean"]})
    return grouped


def histogram_build(dframe, label) -> list:
    """Function builds a histogram for a random image,
    returns a list of arrays for each channel
    9 point of lab. work"""
    try:
        fitrted_df = filter_by_label(dframe, label)
        image = fitrted_df["Absolute path"].sample().values[0]
        image_bgr = cv2.imread(image)
        b, g, r = cv2.split(image_bgr)
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
        hists = [hist_b, hist_g, hist_r]
        return hists
    except:
        logging.error(f"File for histogram was not found: {ex.message}\n{ex.args}\n")


def draw_histogram(hists):
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


if __name__ == "__main__":
    dframe = pd.read_csv("Lab4\csv_files\data.csv")
    draw_histogram(histogram_build(dframe, 0))
