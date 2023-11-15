import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple


def create_output_folder(folder_name: str) -> None:
    """Create new folder"""
    os.makedirs(folder_name, exist_ok=True)

def read_csv_file(file_path: str) -> pd.DataFrame:
    """Read data from csv file and returns dataframe"""
    return pd.read_csv(file_path, header=None, names=['Date', 'Value'])

def convert_to_datetime(data: pd.DataFrame) -> pd.DataFrame:
    """Converts the 'Date' column to datetime format"""
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def split_data_by_weeks(data: pd.DataFrame) -> Tuple[datetime, datetime, pd.DataFrame]:
    """Splits the data into separate weeks."""
    min_date = data['Date'].min()
    max_date = data['Date'].max()

    start_date = min_date
    end_date = min_date + timedelta(days=6 - min_date.weekday())

    while start_date <= max_date:
        week_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
        yield start_date, end_date, week_data
        
        start_date = end_date + timedelta(days=1)
        end_date = start_date + timedelta(days=6)
        if end_date > max_date:
            end_date = max_date

if __name__ == "__main__":
    output_folder = 'script3_files'
    create_output_folder(output_folder)

    file_path = 'csv_files/data.csv'
    data = read_csv_file(file_path)
    data = convert_to_datetime(data)

    for start_date, end_date, week_data in split_data_by_weeks(data):
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        file_name = f"{start_date_str}_{end_date_str}.csv"
        
        file_path = os.path.join(output_folder, file_name)
        week_data.to_csv(file_path, index=False)