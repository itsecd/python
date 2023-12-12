import pandas as pd
import json
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        channels = len(img.getbands())
    return height, width, channels


def get_pixel_count(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        return img.size // img.itemsize
    else:
        print(f"Warning: Unable to read image at path {image_path}")
        return 0


def create_dataframe(csv_path):
    df = pd.read_csv(csv_path, header=None, names=[
                     'absolute_path', 'relative_path', 'class'], sep=',')
    df['label'] = df['class'].factorize()[0]

    class_mapping = {'leopard': 0, 'tiger': 1}
    df['numeric_label'] = df['class'].map(class_mapping)

    df['height'], df['width'], df['depth'] = zip(
        *df['absolute_path'].apply(get_image_dimensions))
    df['pixel_count'] = df['absolute_path'].apply(get_pixel_count)

    result_df = df[['class', 'absolute_path', 'numeric_label',
                    'height', 'width', 'depth', 'pixel_count']].copy()
    result_df.columns = ['class_name', 'absolute_path',
                         'numeric_label', 'height', 'width', 'depth', 'pixel_count']
    return result_df


def compute_statistics(df):
    image_dimensions_stats = df[['height', 'width',
                                 'depth', 'pixel_count']].describe()
    print("Statistics for image sizes and pixel count:")
    print(image_dimensions_stats)

    class_label_stats = df['numeric_label'].value_counts()
    print("\nStatistics for class labels:")
    print(class_label_stats)

    is_balanced = class_label_stats.min() == class_label_stats.max()
    print("\nThe data set is balanced:", is_balanced)


def filter_dataframe(input_df, target_label=None, max_height=None, max_width=None):
    filtered_df = input_df.copy()

    if target_label is not None:
        filtered_df = filtered_df[filtered_df['numeric_label'] == target_label]

    if max_height is not None:
        filtered_df = filtered_df[filtered_df['height'] <= max_height]

    if max_width is not None:
        filtered_df = filtered_df[filtered_df['width'] <= max_width]

    return filtered_df


def group_by_label_and_pixel_count(df):
    grouped_df = df.groupby('numeric_label')['pixel_count'].agg(
        ['min', 'max', 'mean']).reset_index()
    grouped_df.columns = ['numeric_label', 'min_pixel_count',
                          'max_pixel_count', 'mean_pixel_count']
    return grouped_df


def plot_histogram(dataframe, target_label):
    filtered_df = dataframe[dataframe['numeric_label'] == target_label]
    random_image_path = np.random.choice(filtered_df['absolute_path'].values)
    image = cv2.imread(random_image_path)

    b, g, r = cv2.split(image)

    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    plt.figure(figsize=(10, 6))
    plt.title(f'Histogram for Class {target_label}')
    plt.plot(hist_b, color='blue', label='Blue Channel')
    plt.plot(hist_g, color='green', label='Green Channel')
    plt.plot(hist_r, color='red', label='Red Channel')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    return hist_b, hist_g, hist_r


def plot_histogram_matplotlib(hist_b, hist_g, hist_r, target_label):
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


if __name__ == "__main__":
    with open("Lab4/options.json", "r") as options_file:
        options = json.load(options_file)
    df = create_dataframe(options['annotation'])
    print(df)
    compute_statistics(df)

    target_label = options["target_label"]
    max_height = options["max_height"]
    max_width = options["max_width"]

    filtered_by_label_df = filter_dataframe(
        df, target_label=target_label, max_height=None, max_width=None)
    filtered_by_params_df = filter_dataframe(
        df, target_label=target_label, max_height=max_height, max_width=max_width)
    grouped_df = group_by_label_and_pixel_count(df)

    print(
        f"\nFiltered DataFrame for class label {target_label}:\n", filtered_by_label_df)
    print(
        f"\nFiltered DataFrame for class label {target_label}, height <= {max_height}, width <= {max_width}:\n", filtered_by_params_df)
    print("\nGrouped DataFrame by numeric label and pixel count:\n", grouped_df)

    hist_b, hist_g, hist_r = plot_histogram(df, target_label)
    plot_histogram_matplotlib(hist_b, hist_g, hist_r, target_label)
