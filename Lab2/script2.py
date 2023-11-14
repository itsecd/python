import os
import pandas as pd
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

def group_data_by_year(data: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """Groups data by year"""
    grouped = data.groupby(data['Date'].dt.year)
    for year, group in grouped:
        yield year, group

def save_group_to_csv(output_folder: str, year: int, group: pd.DataFrame) -> None:
    """Saves a group of data to a csv file"""
    start_date = group['Date'].min().strftime('%Y%m%d')
    end_date = group['Date'].max().strftime('%Y%m%d')
    file_name = f"{start_date}_{end_date}.csv"
    
    file_path = os.path.join(output_folder, file_name)
    group.to_csv(file_path, index=False, header=False)

if __name__ == "__main__":
    output_folder = 'script2_files'
    create_output_folder(output_folder)

    file_path = 'csv_files/data.csv'
    data = read_csv_file(file_path)
    data = convert_to_datetime(data)

    for year, group in group_data_by_year(data):
        save_group_to_csv(output_folder, year, group)