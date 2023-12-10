import cv2
import pandas as pd
import os

def read_csv_to_dataframe(csv_file):
    # Чтение CSV файла без заголовков столбцов
    return pd.read_csv(csv_file, delimiter=";", names=["Absolute path", "Relative path", "Class"])

def process_dataframe(df):
    df = df.drop("Relative path", axis=1)  # Удаляем столбец "Relative path"
    
    heights = []
    widths = []
    channels = []
    abs_paths = df["Absolute path"]
    counter = 0
    class_labels = {'polar bear': 0, 'brown bear': 1}
    df['Label'] = df['Class'].map(class_labels)
    
    for path in abs_paths:
        if os.path.exists(path):  # Проверяем существование файла по указанному пути
            img = cv2.imread(path)
            if img is not None:
                height, width, channel = img.shape
                heights.append(height)
                widths.append(width)
                channels.append(channel)
                print(counter)
                counter+=1
            else:
                heights.append(None)
                widths.append(None)
                channels.append(None)
        else:
            # Если файл не найден, добавляем значения None
            print(f"Изображение не найдено по пути: {path}")
            heights.append(None)
            widths.append(None)
            channels.append(None)
    
    df['Height'] = heights
    df['Width'] = widths
    df['Channels'] = channels
    
    return df[['Absolute path', 'Class', 'Label', 'Height', 'Width', 'Channels']]

def save_to_csv(df, output_csv):
    df.to_csv(output_csv, index=False, sep=';')  # Сохранить DataFrame в CSV без индексов

# Пример использования функций
def main():
    csv_file = 'C:/Users/zhura/Desktop/paths.csv'  # Имя CSV файла
    output_csv = 'C:/Users/zhura/Desktop/processed_data.csv'  # Имя файла для сохранения обработанных данных
    
    # Чтение данных из CSV файла в DataFrame без заголовков столбцов
    data_frame = read_csv_to_dataframe(csv_file)
    
    # Обработка DataFrame
    processed_df = process_dataframe(data_frame)
    
    # Сохранение в новый CSV файл
    save_to_csv(processed_df, output_csv)
    print(f"Данные сохранены в файл {output_csv}")

if __name__ == "__main__":
    main()