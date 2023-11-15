import os
import pandas as pd


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