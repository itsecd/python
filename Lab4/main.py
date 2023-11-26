import pandas as pd
import cv2
import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

def create_dataframe(data_folder):
    """
    Создает DataFrame, содержащий две колонки: 'Class' и 'Path'.

    Parameters:
    - data_folder (str): Путь к папке с данными.

    Returns:
    - pd.DataFrame: DataFrame с информацией о классах и путях к изображениям.
    """
    data = {'Class': [], 'Path': []}

    for class_label in os.listdir(data_folder):
        class_folder = os.path.join(data_folder, class_label)
        if os.path.isdir(class_folder):
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                data['Class'].append(class_label)
                data['Path'].append(file_path)

    return pd.DataFrame(data)

def add_numeric_label_column(df):
    """
    Добавляет числовой столбец 'NumericLabel' в DataFrame.

    Parameters:
    - df (pd.DataFrame): Исходный DataFrame.

    Returns:
    - pd.DataFrame: DataFrame с добавленным столбцом 'NumericLabel'.
    """
    df['NumericLabel'] = df['Class'].apply(lambda x: 0 if x == 'cat' else 1)
    return df

def add_image_dimensions_columns(df):
    """
    Добавляет столбцы 'Height', 'Width' и 'Depth' в DataFrame, содержащие информацию о размерах изображений.

    Parameters:
    - df (pd.DataFrame): Исходный DataFrame.

    Returns:
    - pd.DataFrame: DataFrame с добавленными столбцами.
    """
    df['Height'] = df['Path'].apply(lambda x: get_image_dimensions(x)[0])
    df['Width'] = df['Path'].apply(lambda x: get_image_dimensions(x)[1])
    df['Depth'] = df['Path'].apply(lambda x: get_image_dimensions(x)[2])
    return df

def get_image_dimensions(image_path):
    """
    Получает размеры изображения (высоту, ширину и глубину) с использованием библиотеки Pillow.

    Parameters:
    - image_path (str): Путь к изображению.

    Returns:
    - tuple: Кортеж с высотой, шириной и глубиной изображения.
    """
    with Image.open(image_path) as img:
        return img.height, img.width, len(img.getbands())

def calculate_statistics(df):
    """
    Вычисляет статистическую информацию о размерах изображений и метках класса.

    Parameters:
    - df (pd.DataFrame): Исходный DataFrame.

    Returns:
    - pd.DataFrame: DataFrame со статистикой.
    """
    statistics = df.groupby('Class').agg({
        'Height': ['mean', 'min', 'max'],
        'Width': ['mean', 'min', 'max'],
        'Depth': ['mean', 'min', 'max'],
        'NumericLabel': 'count'
    })
    return statistics

def filter_dataframe_by_class(df, class_label):
    """
    Фильтрует DataFrame по метке класса.

    Parameters:
    - df (pd.DataFrame): Исходный DataFrame.
    - class_label (str): Метка класса для фильтрации.

    Returns:
    - pd.DataFrame: Отфильтрованный DataFrame.
    """
    filtered_df = df[df['Class'] == class_label]
    return filtered_df

def filter_dataframe_by_dimensions(df, class_label, max_width, max_height):
    """
    Фильтрует DataFrame по размерам изображений и метке класса.

    Parameters:
    - df (pd.DataFrame): Исходный DataFrame.
    - class_label (str): Метка класса для фильтрации.
    - max_width (int): Максимальная ширина изображения.
    - max_height (int): Максимальная высота изображения.

    Returns:
    - pd.DataFrame: Отфильтрованный DataFrame.
    """
    filtered_df = df[(df['Class'] == class_label) & (df['Width'] <= max_width) & (df['Height'] <= max_height)]
    return filtered_df

def calculate_pixel_statistics(df):
    """
    Группирует DataFrame по метке класса с вычислением статистики по количеству пикселей.

    Parameters:
    - df (pd.DataFrame): Исходный DataFrame.

    Returns:
    - pd.DataFrame: DataFrame со статистикой по количеству пикселей.
    """
    df['PixelCount'] = df.apply(lambda row: calculate_image_pixel_count(row['Height'], row['Width'], row['Depth']), axis=1)
    pixel_statistics = df.groupby('Class')['PixelCount'].agg(['mean', 'min', 'max'])
    return pixel_statistics

def calculate_image_pixel_count(height, width, depth):
    """
    Вычисляет количество пикселей на изображении.

    Parameters:
    - height (int): Высота изображения.
    - width (int): Ширина изображения.
    - depth (int): Глубина изображения (количество каналов).

    Returns:
    - int: Количество пикселей.
    """
    return height * width * depth

def generate_histogram(df, class_label):
    """
    Генерирует гистограмму для случайного изображения из DataFrame.

    Parameters:
    - df (pd.DataFrame): Исходный DataFrame.
    - class_label (str): Метка класса изображения.

    Returns:
    - list: Список массивов значений гистограммы для каждого канала.
    """
    # Выбор случайного изображения из DataFrame для указанного класса
    random_image = df[df['Class'] == class_label].sample(1)

    # Получение пути к изображению
    image_path = random_image.iloc[0]['Path']

    # Чтение изображения с использованием OpenCV
    image = cv2.imread(image_path)

    # Построение гистограммы для каждого канала изображения
    histogram_data = []
    for i in range(image.shape[2]):
        channel_data = image[:, :, i].ravel()
        histogram_data.append(channel_data)

    return histogram_data

def plot_histogram(histogram_data):
    """
    Отрисовывает гистограмму с использованием matplotlib.

    Parameters:
    - histogram_data (list): Список массивов значений гистограммы для каждого канала.
    """
    plt.figure(figsize=(10, 5))
    colors = ['red', 'green', 'blue']

    for i, data in enumerate(histogram_data):
        plt.hist(data, bins=256, range=[0, 256], color=colors[i], alpha=0.7, label=f'Channel {i + 1}')

    plt.title('Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Пример использования функций
    data_folder = "путь_к_вашим_данным"
    df = create_dataframe(data_folder)
    df = add_numeric_label_column(df)
    df = add_image_dimensions_columns(df)

    # Вывод статистики
    print("Statistics:")
    print(calculate_statistics(df))

    # Фильтрация DataFrame
    filtered_df_class = filter_dataframe_by_class(df, 'cat')
    filtered_df_dimensions = filter_dataframe_by_dimensions(df, 'cat', 500, 500)
    print("\nFiltered DataFrame by Class:")
    print(filtered_df_class.head())
    print("\nFiltered DataFrame by Dimensions:")
    print(filtered_df_dimensions.head())

    # Статистика по количеству пикселей
    pixel_statistics = calculate_pixel_statistics(df)
    print("\nPixel Statistics:")
    print(pixel_statistics)

    # Генерация и отрисовка гистограммы
    histogram_data = generate_histogram(df, 'cat')
    plot_histogram(histogram_data)
