import pandas as pd
import matplotlib.pyplot as plt
from csv_file_processing import preparing_dataframe


def create_graph(file_path):
    resulting_dataframe = preparing_dataframe(file_path)
    resulting_dataframe['Date'] = pd.to_datetime(resulting_dataframe['Date'])

    plt.figure(figsize=(10, 6))
    plt.plot(resulting_dataframe['Date'], resulting_dataframe['Value'], color='green')

    plt.xlabel('Дата')
    plt.ylabel('Значение курса')
    plt.title('Изменение курса за весь период')

    plt.tight_layout()
    plt.show()


def create_monthly_graph(dataframe, month):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    filtered_data = dataframe[dataframe['Date'].dt.month == month]

    plt.figure(figsize=(10, 6)) 
    plt.plot(filtered_data['Date'], filtered_data['Value'], label='Значение курса', color='blue') 
    plt.axhline(y=filtered_data['Value'].median(), color='red', linestyle='--', label='Медиана') 
    plt.axhline(y=filtered_data['Value'].mean(), color='green', linestyle='--', label='Среднее значение')

    plt.xlabel('Дата')
    plt.ylabel('Значение курса')
    plt.title(f'Изменение курса в месяце {month}')
    plt.legend()

    plt.tight_layout()
    plt.show()
