import pandas as pd
import matplotlib.pyplot as plt

def plot_weather_statistics(df, start_date, end_date):

    df['Дата'] = pd.to_datetime(df['Дата'], format='%Y-%m-%d')
    df['Ветер'] = df['Ветер'].str.extract('(\d+)', expand=False)
    df['Ветер'] = pd.to_numeric(df['Ветер'], errors='coerce')


    filtered_data = df[(df['Дата'] >= start_date) & (df['Дата'] <= end_date)]


    if filtered_data.empty:
        print(f'Нет данных за период {start_date} - {end_date}.')
        return


    plt.figure(figsize=(15, 8))


    plt.subplot(2, 2, 1)
    plt.plot(filtered_data['Дата'], filtered_data['Температура'], label='Температура (C)')
    plt.axhline(y=filtered_data['Температура'].median(), color='r', linestyle='--', label='Медиана')
    plt.axhline(y=filtered_data['Температура'].mean(), color='g', linestyle='--', label='Среднее значение')
    plt.xlabel('Дата')
    plt.ylabel('Температура')
    plt.title('График изменения температуры')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(filtered_data['Дата'], filtered_data['Давление'], label='Давление')
    plt.axhline(y=filtered_data['Давление'].median(), color='r', linestyle='--', label='Медиана')
    plt.axhline(y=filtered_data['Давление'].mean(), color='g', linestyle='--', label='Среднее значение')
    plt.xlabel('Дата')
    plt.ylabel('Давление')
    plt.title('График изменения давления')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(filtered_data['Дата'], filtered_data['Ветер'], label='Скорость ветра')
    plt.axhline(y=filtered_data['Ветер'].median(), color='r', linestyle='--', label='Медиана')
    plt.axhline(y=filtered_data['Ветер'].mean(), color='g', linestyle='--', label='Среднее значение')
    plt.xlabel('Дата')
    plt.ylabel('Скорость ветра')
    plt.title('График изменения скорости ветра')
    plt.legend()

    plt.tight_layout()
    plt.show()

file_path = 'Lab4\dataset'

df = pd.read_csv('dataset//data.csv')

df.columns = ['Дата', 'Температура', 'Давление', 'Ветер']


if df.isnull().values.any():

    df = df.dropna() 

plot_weather_statistics(df, start_date='01/10/2008', end_date='31/12/2009')
