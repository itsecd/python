import pandas as pd
import json
from PIL import Image


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        channels = len(img.getbands())
    return height, width, channels


def create_dataframe(csv_path):
    df = pd.read_csv(csv_path, header=None, names=[
                     'absolute_path', 'relative_path', 'class'], sep=',')
    df['label'] = df['class'].factorize()[0]

    class_mapping = {'leopard': 0, 'tiger': 1}
    df['numeric_label'] = df['class'].map(class_mapping)

    df['height'], df['width'], df['depth'] = zip(
        *df['absolute_path'].apply(get_image_dimensions))

    result_df = df[['class', 'absolute_path',
                    'numeric_label', 'height', 'width', 'depth']].copy()
    result_df.columns = ['class_name', 'absolute_path',
                         'numeric_label', 'height', 'width', 'depth']
    return result_df


def compute_statistics(df):
    image_dimensions_stats = df[['height', 'width', 'depth']].describe()
    print("Statistics for image sizes:")
    print(image_dimensions_stats)

    class_label_stats = df['numeric_label'].value_counts()
    print("\nStatistics for class labels:")
    print(class_label_stats)

    is_balanced = class_label_stats.min() == class_label_stats.max()
    print("\nThe data set is balanced:", is_balanced)


def filter_dataframe_by_label(input_df, target_label):
    filtered_df = input_df[input_df['numeric_label'] == target_label].copy()
    return filtered_df


if __name__ == "__main__":
    with open("Lab4/options.json", "r") as options_file:
        options = json.load(options_file)
    df = create_dataframe(options['annotation'])
    print(df)
    compute_statistics(df)
    target_label = 0
    filtered_df = filter_dataframe_by_label(df, target_label)

    print(
        f"\nFiltered DataFrame for class label {target_label}:\n", filtered_df)
