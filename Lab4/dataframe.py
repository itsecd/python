import argparse
import logging
import pandas as pd
from PIL import Image
from histograms import plot_histogram


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_dataset(cat_annotation_file : str, dog_annotation_file : str) -> None:
    """
    func work with dataset(do dataframe and sort dataframe)

    Parameters
    ----------
    cat_annotation_file : str
    dog_annotation_file : str
    """
    cat_df = pd.read_csv(cat_annotation_file)
    dog_df = pd.read_csv(dog_annotation_file)

    cat_df['class'] = 'cat'
    dog_df['class'] = 'dog'

    cat_df.rename(columns={'the text name of the class': 'class_cat'}, inplace=True)
    dog_df.rename(columns={'the text name of the class': 'class_dog'}, inplace=True)

    df = pd.concat([cat_df, dog_df], ignore_index=True)
    df['label'] = df['class'].astype('category').cat.codes

    df['height'] = df['The absolute path'].apply(lambda x: Image.open(x).height)
    df['width'] = df['The absolute path'].apply(lambda x: Image.open(x).width)
    df['channels'] = df['The absolute path'].apply(lambda x: len(Image.open(x).split()))

    df.rename(columns={'The absolute path': 'absolute_path'}, inplace=True)
    df = df[['class', 'absolute_path', 'label', 'height', 'width', 'channels']]

    df['pixel_count'] = df.apply(lambda row: Image.open(row['absolute_path']).size[0] * Image.open(row['absolute_path']).size[1], axis=1)

    logger.info("\nStatistical information for image sizes:")
    logger.info(df[['width', 'height', 'channels']].describe())

    logger.info("\nStatistical information for class labels:")
    logger.info(df['label'].value_counts())

    class_balance = df['label'].value_counts().tolist()
    is_balanced = all(balance == class_balance[0] for balance in class_balance[1:])

    if is_balanced:
        logger.info("The dataset is balanced.")
    else:
        logger.info("The dataset is not balanced.")

    return df


def filter_by_class(df: pd.DataFrame, class_label: str) -> pd.DataFrame:
    """
    filter dataframe by class

    Parameters
    ----------
    df : pd.DataFrame
    class_label : str
    """
    filtered_df = df[df['class'] == class_label].reset_index(drop=True)
    return filtered_df


def filter_by_size_and_class(df: pd.DataFrame, class_label: str, max_width: int, max_height: int) -> pd.DataFrame:
    """
    filter DataFrame by class, width and height.

    Parameters
    ----------
    df : pd.DataFrame
    class_label: str
    max_width : int
    max_height : int
    """
    filtered_df = df[(df['class'] == class_label) & (df['width'] <= max_width) & (df['height'] <= max_height)].reset_index(drop=True)
    return filtered_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Dataset Analysis and Filtering')
    parser.add_argument('cat_annotation_file', type=str, help='Path to the cat annotation file')
    parser.add_argument('dog_annotation_file', type=str, help='Path to the dog annotation file')
    parser.add_argument('--filter-class', type=str, help='Filter by class label')
    parser.add_argument('--filter-width', type=int, help='Max width for filtering')
    parser.add_argument('--filter-height', type=int, help='Max height for filtering')
    args = parser.parse_args()

    df = analyze_dataset(args.cat_annotation_file, args.dog_annotation_file)

    if args.filter_class:
        filtered_df = filter_by_class(df, class_label=args.filter_class)
        logger.info("\nDataFrame filtered by class:")
        logger.info(filtered_df)

    if args.filter_width and args.filter_height:
        filtered_size_df = filter_by_size_and_class(df, class_label=args.filter_class,
                                                     max_width=args.filter_width, max_height=args.filter_height)
        logger.info("\nDataFrame filtered by size and class:")
        logger.info(filtered_size_df)

    grouped_df = df.groupby('class')['pixel_count'].agg(['min', 'max', 'mean']).reset_index()
    logger.info("\nStatistical information for pixel count:")
    logger.info(grouped_df)

    random_image = df.sample(1).iloc[0]
    image_path = random_image['absolute_path']
    print("\n")
    plot_histogram(image_path)