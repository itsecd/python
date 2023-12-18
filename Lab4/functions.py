import pandas as pd
from PIL import Image
import cv2
import logging


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


def get_pixel_count(image_path: str) -> int:
    """
    Get the pixel count of an image. 
    Accepts entry: image_path: The path to the image. 
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


def create_dataframe(csv_path: str) -> pd.DataFrame:
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

    result_df['pixel_count'] = result_df['absolute_path'].apply(
        lambda x: get_pixel_count(x))

    return result_df


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics for the DataFrame. Parameters: df: Input DataFrame"""
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


def filter_dataframe(input_df: pd.DataFrame, target_label: int = None, max_height: int = None, max_width: int = None) -> pd.DataFrame:
    """
    Filter the DataFrame based on specified criteria.

    Parameters:
        input_df: Input DataFrame.
        target_label: Numeric label to filter on.
        max_height: Maximum height for filtering.
        max_width: Maximum width for filtering.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    try:
        filtered_df = input_df.copy()

        if target_label is not None:
            filtered_df = filtered_df[filtered_df['label'] == target_label]
            logging.info(
                f"Filtered DataFrame for target_label {target_label}.")

        if max_height is not None:
            filtered_df = filtered_df[filtered_df['height'] <= max_height]
            logging.info(f"Filtered DataFrame for max_height <= {max_height}.")

        if max_width is not None:
            filtered_df = filtered_df[filtered_df['width'] <= max_width]
            logging.info(f"Filtered DataFrame for max_width <= {max_width}.")

        return filtered_df
    except Exception as e:
        logging.error(f"Error in filter_dataframe: {e}")
        return pd.DataFrame()


def group_by_label_and_pixel_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group the DataFrame by numeric label and compute pixel count statistics.

    Parameters:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Grouped DataFrame.
    """
    try:
        grouped_df = df.groupby(['label', 'class'])['pixel_count'].agg(
            ['min', 'max', 'mean']).reset_index()
        grouped_df.columns = [
            'label', 'class', 'min_pixel_count', 'max_pixel_count', 'mean_pixel_count']

        logging.info(
            "Grouped DataFrame by numeric label and pixel count.")
        return grouped_df
    except Exception as e:
        logging.error(
            f"Error in group_by_label_and_pixel_count: {e}")
        return pd.DataFrame()
