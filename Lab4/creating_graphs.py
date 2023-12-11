import logging
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from csv_file_processing import preparing_dataframe


logging.basicConfig(level=logging.INFO)


def create_graph(file_path: str) -> None:
    """Create a graph displaying the change in currency value over the entire period."""
    try:
        resulting_dataframe = preparing_dataframe(file_path)
        resulting_dataframe['Date'] = pd.to_datetime(resulting_dataframe['Date'])

        plt.figure(figsize=(10, 6))
        plt.plot(resulting_dataframe['Date'], resulting_dataframe['Value'], color='green')

        plt.xlabel('Дата')
        plt.ylabel('Значение курса')
        plt.title('Изменение курса за весь период')

        plt.tight_layout()
        plt.show()
    except Exception as ex:
        logging.error(f"can't show figure: {ex}\n{ex.args}\n")


def create_monthly_graph(dataframe: pd.DataFrame, month: Optional[str] = None) -> None:
    """Create a graph displaying the change in currency value for a specific month."""
    try:
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        dataframe['Month'] = dataframe['Date'].dt.to_period('M')
    except Exception as ex:
        logging.error(f"Can't convert date to datetime: {ex}\n{ex.args}\n")

    try:
        filtered_data = dataframe[dataframe['Month'] == month]
        
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_data['Date'], filtered_data['Value'], label='Изменение курса')
        
        monthly_mean = filtered_data['Value'].mean()
        monthly_median = filtered_data['Value'].median()
        
        plt.axhline(y=monthly_mean, color='r', linestyle='--', label=f'Среднее значение: {monthly_mean:.2f}')
        plt.axhline(y=monthly_median, color='g', linestyle='-.', label=f'Медиана: {monthly_median:.2f}')
        
        plt.title(f'Изменение курса для месяца {month}')
        plt.xlabel('Дата')
        plt.ylabel('Курс доллара')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as ex:
        logging.error(f"Can't create graph for the month: {ex}\n{ex.args}\n")