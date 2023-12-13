import cv2
import pandas as pd

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("Relative path", axis=1, errors='ignore')
    heights = []
    widths = []
    channels = []
    abs_paths = df["Absolute path"]
    class_labels = {'brown bear': 0, 'polar bear': 1}
    df['Label'] = df['Class'].map(class_labels)
    for path in abs_paths:
        try:
            image = cv2.imread(path)
            if image is not None:
                height, width, channel = image.shape
                heights.append(height)
                widths.append(width)
                channels.append(channel)
            else:
                heights.append(None)
                widths.append(None)
                channels.append(None)
        except Exception as e:
            print(f"Ошибка при обработке изображения {absolute_path}: {str(e)}")

    # Модификация фрейма данных
    df["Высота"] = heights
    df["Ширина"] = widths
    df["Каналы"] = channels

    return df[['Absolute path', 'Class', 'Label', 'Height', 'Width', 'Channels']]