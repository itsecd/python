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