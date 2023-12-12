import pandas as pd
import json
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from read_and_write import open_csv, save_csv


logging.basicConfig(level=logging.INFO)


def get_image_dimensions(image_path: str, dimension: str = 'all') -> tuple[int, int, int]:
    """
    Get the dimensions of an image.
    Accepts entry: The path to the image.
    Returns: Image height, width, and channels
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            channels = len(img.getbands())
        logging.info(
            f"Image dimensions for {image_path}: {width}x{height}, Channels: {channels}")
        if dimension == 'height':
            return height
        elif dimension == 'width':
            return width
        elif dimension == 'depth':
            return channels
        else:
            return height, width, channels
    except Exception as e:
        logging.warning(f"Error in get_image_dimensions for {image_path}: {e}")
        return 0, 0, 0


def create_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Function for creating a DataFrame from a CSV file.

    Parameters:
    - csv_path: The path to the CSV file.
    Returns:
    - pd.DataFrame: The created DataFrame with specified columns.
    """
    df = pd.read_csv(csv_path, header=None, names=[
                     'absolute_path', 'relative_path', 'class'], sep=',')

    result_df = df[['class', 'absolute_path']].copy()
    result_df.columns = ['class', 'absolute_path']

    result_df['label'] = pd.factorize(result_df['class'])[0]

    result_df['height'] = result_df['absolute_path'].apply(
        lambda x: get_image_dimensions(x, dimension='height'))
    result_df['width'] = result_df['absolute_path'].apply(
        lambda x: get_image_dimensions(x, dimension='width'))
    result_df['depth'] = result_df['absolute_path'].apply(
        lambda x: get_image_dimensions(x, dimension='depth'))

    return result_df


def get_pixel_count(image_path: str) -> int:
    """
    Get the pixel count of an image. 
    Accepts entry: image_path (str): The path to the image. 
    Returns: int: Pixel count
    """
    try:
        img = cv2.imread(image_path)
        if img is not None:
            pixel_count = img.size // img.itemsize
            logging.info(f"Pixel count for {image_path}: {pixel_count}")
            return pixel_count
        else:
            logging.warning(f"Unable to read image at path {image_path}")
            return 0
    except Exception as e:
        logging.warning(f"Error in get_pixel_count for {image_path}: {e}")
        return 0


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics for the DataFrame. Parameters: df (pd.DataFrame): Input DataFrame"""
    try:
        image_dimensions_stats = df[['height', 'width', 'depth']].describe()
        logging.info("Statistics for image sizes:")
        logging.info(image_dimensions_stats)

        class_label_stats = df['class'].value_counts()
        logging.info("Statistics for class labels:")
        logging.info(class_label_stats)

        is_balanced = class_label_stats.min() == class_label_stats.max()
        logging.info("The data set is balanced: %s", is_balanced)

        return pd.concat([image_dimensions_stats, class_label_stats], axis=1)

    except Exception as e:
        logging.error(f"Error in compute_statistics: {e}")
        return pd.DataFrame()


# def filter_dataframe(input_df: pd.DataFrame, target_label: int = None, max_height: int = None, max_width: int = None) -> pd.DataFrame:
#     """
#     Filter the DataFrame based on specified criteria.

#     Parameters:
#         input_df (pd.DataFrame): Input DataFrame.
#         target_label (int): Numeric label to filter on.
#         max_height (int): Maximum height for filtering.
#         max_width (int): Maximum width for filtering.

#     Returns:
#         pd.DataFrame: Filtered DataFrame.
#     """
#     try:
#         filtered_df = input_df.copy()

#         if target_label is not None:
#             filtered_df = filtered_df[filtered_df['numeric_label']
#                                       == target_label]
#             logging.info(
#                 f"Filtered DataFrame for target_label {target_label}.")

#         if max_height is not None:
#             filtered_df = filtered_df[filtered_df['height'] <= max_height]
#             logging.info(f"Filtered DataFrame for max_height <= {max_height}.")

#         if max_width is not None:
#             filtered_df = filtered_df[filtered_df['width'] <= max_width]
#             logging.info(f"Filtered DataFrame for max_width <= {max_width}.")

#         return filtered_df
#     except Exception as e:
#         logging.error(f"Error in filter_dataframe: {e}")
#         return pd.DataFrame()


# def group_by_label_and_pixel_count(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Group the DataFrame by numeric label and compute pixel count statistics.

#     Parameters:
#         df (pd.DataFrame): Input DataFrame.

#     Returns:
#         pd.DataFrame: Grouped DataFrame.
#     """
#     try:
#         grouped_df = df.groupby('numeric_label')['pixel_count'].agg(
#             ['min', 'max', 'mean']).reset_index()
#         grouped_df.columns = [
#             'numeric_label', 'min_pixel_count', 'max_pixel_count', 'mean_pixel_count']

#         logging.info("Grouped DataFrame by numeric label and pixel count.")
#         return grouped_df
#     except Exception as e:
#         logging.error(f"Error in group_by_label_and_pixel_count: {e}")
#         return pd.DataFrame()


# def plot_histogram(dataframe: pd.DataFrame, target_label: int) -> tuple:
#     """
#     Plot histograms for the RGB channels of images in the DataFrame.

#     Parameters:
#         dataframe (pd.DataFrame): Input DataFrame.
#         target_label (int): Numeric label for filtering.

#     Returns:
#         tuple: Histograms for the Blue, Green, and Red channels.
#     """
#     try:
#         filtered_df = dataframe[dataframe['numeric_label'] == target_label]
#         random_image_path = np.random.choice(
#             filtered_df['absolute_path'].values)
#         image = cv2.imread(random_image_path)

#         b, g, r = cv2.split(image)

#         hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
#         hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
#         hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

#         plt.figure(figsize=(10, 6))
#         plt.title(f'Histogram for Class {target_label}')
#         plt.plot(hist_b, color='blue', label='Blue Channel')
#         plt.plot(hist_g, color='green', label='Green Channel')
#         plt.plot(hist_r, color='red', label='Red Channel')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')
#         plt.legend()
#         plt.show()

#         logging.info(f"Histogram plotted for Class {target_label}.")
#         return hist_b, hist_g, hist_r
#     except Exception as e:
#         logging.error(f"Error in plot_histogram: {e}")
#         return None, None, None


# def plot_histogram_matplotlib(hist_b: np.ndarray, hist_g: np.ndarray, hist_r: np.ndarray, target_label: int) -> None:
#     """
#     Plot histograms for the RGB channels using Matplotlib.

#     Parameters:
#         hist_b (np.ndarray): Histogram for the Blue channel.
#         hist_g (np.ndarray): Histogram for the Green channel.
#         hist_r (np.ndarray): Histogram for the Red channel.
#         target_label (int): Numeric label for the class.
#     """
#     try:
#         plt.figure(figsize=(15, 8))

#         plt.subplot(3, 1, 1)
#         plt.plot(hist_b, color='blue')
#         plt.title(f'Blue Channel Histogram for Class {target_label}')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')

#         plt.subplot(3, 1, 2)
#         plt.plot(hist_g, color='green')
#         plt.title(f'Green Channel Histogram for Class {target_label}')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')

#         plt.subplot(3, 1, 3)
#         plt.plot(hist_r, color='red')
#         plt.title(f'Red Channel Histogram for Class {target_label}')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')

#         plt.tight_layout()
#         plt.show()

#         logging.info(f"Histograms plotted for Class {target_label}.")
#     except Exception as e:
#         logging.error(f"Error in plot_histogram_matplotlib: {e}")


# if __name__ == "__main__":
#     df = create_dataframe('annotation.csv')
#     print(df)
#     compute_statistics(df)

#     target_label = options["target_label"]
#     max_height = options["max_height"]
#     max_width = options["max_width"]

#     filtered_by_label_df = filter_dataframe(
#         df, target_label=target_label, max_height=None, max_width=None)
#     filtered_by_params_df = filter_dataframe(
#         df, target_label=target_label, max_height=max_height, max_width=max_width)
#     grouped_df = group_by_label_and_pixel_count(df)

#     print(
#         f"\nFiltered DataFrame for class label {target_label}:\n", filtered_by_label_df)
#     print(
#         f"\nFiltered DataFrame for class label {target_label}, height <= {max_height}, width <= {max_width}:\n", filtered_by_params_df)
#     print("\nGrouped DataFrame by numeric label and pixel count:\n", grouped_df)

#     hist_b, hist_g, hist_r = plot_histogram(df, target_label)
#     plot_histogram_matplotlib(hist_b, hist_g, hist_r, target_label)
