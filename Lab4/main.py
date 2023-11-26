import pandas as pd
import cv2
import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

def create_dataframe(data_folder):
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
    df['NumericLabel'] = df['Class'].apply(lambda x: 0 if x == 'cat' else 1)
    return df

def add_image_dimensions_columns(df):
    df['Height'] = df['Path'].apply(lambda x: get_image_dimensions(x)[0])
    df['Width'] = df['Path'].apply(lambda x: get_image_dimensions(x)[1])
    df['Depth'] = df['Path'].apply(lambda x: get_image_dimensions(x)[2])
    return df

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.height, img.width, len(img.getbands())

def calculate_statistics(df):
    statistics = df.groupby('Class').agg({
        'Height': ['mean', 'min', 'max'],
        'Width': ['mean', 'min', 'max'],
        'Depth': ['mean', 'min', 'max'],
        'NumericLabel': 'count'
    })
    return statistics

def filter_dataframe_by_class(df, class_label):
    filtered_df = df[df['Class'] == class_label]
    return filtered_df

def filter_dataframe_by_dimensions(df, class_label, max_width, max_height):
    filtered_df = df[(df['Class'] == class_label) & (df['Width'] <= max_width) & (df['Height'] <= max_height)]
    return filtered_df

def calculate_pixel_statistics(df):
    df['PixelCount'] = df.apply(lambda row: calculate_image_pixel_count(row['Height'], row['Width'], row['Depth']), axis=1)
    pixel_statistics = df.groupby('Class')['PixelCount'].agg(['mean', 'min', 'max'])
    return pixel_statistics

def generate_histogram(df, class_label):
    # Выбор случайного изображения из DataFrame для указанного класса
    random_image = df[df['Class'] == class_label].sample(1)

    # Получение пути к изображению
    image_path = random_image.iloc[0]['ImagePath']

    # Чтение изображения с использованием OpenCV
    image = cv2.imread(image_path)

    # Построение гистограммы для каждого канала изображения
    histogram_data = []
    for i in range(image.shape[2]):
        channel_data = image[:, :, i].ravel()
        histogram_data.append(channel_data)

    return histogram_data

def plot_histogram(histogram_data):
    # Отрисовка гистограммы с использованием matplotlib
    plt.figure(figsize=(10, 5))
    colors = ['red', 'green', 'blue']

    for i, data in enumerate(histogram_data):
        plt.hist(data, bins=256, range=[0, 256], color=colors[i], alpha=0.7, label=f'Channel {i + 1}')

    plt.title('Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def calculate_image_pixel_count(height, width, depth):
    return height * width * depth

# Пример использования функций
df = create_dataframe()
df = add_numeric_label_column(df)
df = add_image_dimensions_columns(df)
statistics = calculate_statistics(df)
filtered_df = filter_dataframe_by_class(df, 'cat')
filtered_df_by_dimensions = filter_dataframe_by_dimensions(df, 'cat', 500, 500)
pixel_statistics = calculate_pixel_statistics(df)
histogram_data = generate_histogram(df, 'cat')
plot_histogram(histogram_data)
