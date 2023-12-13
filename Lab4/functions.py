import cv2
import pandas as pd


def process_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.drop("Relative path", axis=1, errors='ignore')
    heights = []
    widths = []
    channels = []
    abs_paths = dataframe["Absolute path"]
    class_labels = {'brown bear': 0, 'polar bear': 1}
    dataframe['Label'] = dataframe['Class'].map(class_labels)
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
            print(f"Ошибка при обработке изображения: {str(e)}")

    # Модификация фрейма данных
    dataframe["Высота"] = heights
    dataframe["Ширина"] = widths
    dataframe["Каналы"] = channels

    return dataframe[['Absolute path', 'Class', 'Height', 'Width', 'Channels']]

def filter_by_label(dataframe: pd.DataFrame, class_label) -> pd.DataFrame:
    return dataframe[dataframe['Class'] == class_label]

def filter_by_parameters(dataframe: pd.DataFrame, class_label, max_height, max_width) -> pd.DataFrame:
    return dataframe[(dataframe['Class'] == class_label) &
                     (dataframe['Height'] <= max_height) &
                     (dataframe['Width'] <= max_width)]

