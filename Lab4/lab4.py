import pandas as pd
import matplotlib.pyplot as plt
import argparse
from typing import Union


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует данные в DataFrame, обрабатывая столбцы 'Дата' и 'Ветер'.

    Parameters:
    - df: pandas.DataFrame, входной DataFrame.

    """
    df['Дата'] = pd.to_datetime(df['Дата'], format='%Y-%m-%d')
    df['Ветер'] = df['Ветер'].str.extract('(\d+)', expand=False)
    df['Ветер'] = pd.to_numeric(df['Ветер'], errors='coerce')
    return df

def filter_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Фильтрует данные в DataFrame по указанному периоду.

    Parameters:
    - df: pandas.DataFrame, входной DataFrame.
    - start_date: str, начальная дата в формате 'DD/MM/YYYY'.
    - end_date: str, конечная дата в формате 'DD/MM/YYYY'.

    """
    filtered_data = df[(df['Дата'] >= start_date) & (df['Дата'] <= end_date)]
    return filtered_data

def plot_temperature_statistics(data: pd.DataFrame) -> None:
    """
    Строит график изменения температуры, медианы и среднего значения.

    Parameters:
    - data: pandas.DataFrame, входной DataFrame.

    """
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    plt.plot(data['Дата'], data['Температура'], label='Температура (C)')
    plt.axhline(y=data['Температура'].median(), color='r', linestyle='--', label='Медиана')
    plt.axhline(y=data['Температура'].mean(), color='g', linestyle='--', label='Среднее значение')
    plt.xlabel('Дата')
    plt.ylabel('Температура')
    plt.title('График изменения температуры')
    plt.legend()

def plot_pressure_statistics(data: pd.DataFrame) -> None:
    """
    Строит график изменения давления, медианы и среднего значения.

    Parameters:
    - data: pandas.DataFrame, входной DataFrame.

    """
    plt.subplot(2, 2, 2)
    plt.plot(data['Дата'], data['Давление'], label='Давление')
    plt.axhline(y=data['Давление'].median(), color='r', linestyle='--', label='Медиана')
    plt.axhline(y=data['Давление'].mean(), color='g', linestyle='--', label='Среднее значение')
    plt.xlabel('Дата')
    plt.ylabel('Давление')
    plt.title('График изменения давления')
    plt.legend()

def plot_wind_speed_statistics(data: pd.DataFrame) -> None:
    """
    Строит график изменения скорости ветра, медианы и среднего значения.

    Parameters:
    - data: pandas.DataFrame, входной DataFrame.

    """
    plt.subplot(2, 2, 3)
    plt.plot(data['Дата'], data['Ветер'], label='Скорость ветра')
    plt.axhline(y=data['Ветер'].median(), color='r', linestyle='--', label='Медиана')
    plt.axhline(y=data['Ветер'].mean(), color='g', linestyle='--', label='Среднее значение')
    plt.xlabel('Дата')
    plt.ylabel('Скорость ветра')
    plt.title('График изменения скорости ветра')
    plt.legend()

def save_and_show_plots(output_file: str = 'weather_statistics.png') -> None:
    """
    Сохраняет и отображает графики.

    Parameters:
    - output_file: str, имя файла для сохранения графиков.

    """
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f'Графики сохранены в файл: {output_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot weather statistics for a given period.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file with weather data')
    parser.add_argument('start_date', type=str, help='Start date in the format DD/MM/YYYY')
    parser.add_argument('end_date', type=str, help='End date in the format DD/MM/YYYY')
    parser.add_argument('--output_file', type=str, default='weather_statistics.png', help='Output file for saving the plots')
    
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)
    df.columns = ['Дата', 'Температура', 'Давление', 'Ветер']

    if df.isnull().values.any():
        df = df.dropna()  

    df = preprocess_data(df)
    filtered_data = filter_data(df, start_date=args.start_date, end_date=args.end_date)

    if not filtered_data.empty:
        plot_temperature_statistics(filtered_data)
        plot_pressure_statistics(filtered_data)
        plot_wind_speed_statistics(filtered_data)
        save_and_show_plots(output_file=args.output_file)
