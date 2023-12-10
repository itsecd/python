import pandas as pd
import json
from PIL import Image

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        channels = len(img.getbands())
    return height, width, channels

def create_dataframe(csv_path):
    df = pd.read_csv(csv_path, header=None, names=['absolute_path', 'relative_path', 'query'], sep=',')
    df['label'] = df['query'].factorize()[0]

    class_mapping = {'leopard': 0, 'tiger': 1}
    df['numeric_label'] = df['query'].map(class_mapping)

    df['height'], df['width'], df['depth'] = zip(*df['absolute_path'].apply(get_image_dimensions))

    result_df = df[['query', 'absolute_path', 'numeric_label', 'height', 'width', 'depth']].copy()
    result_df.columns = ['class_name', 'absolute_path', 'numeric_label', 'height', 'width', 'depth']
    return result_df

if __name__ == "__main__":
    with open("Lab4/options.json", "r") as options_file:
        options = json.load(options_file)
    df = create_dataframe(options['annotation'])
    print(df)