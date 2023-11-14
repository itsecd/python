import os
import pandas as pd
from typing import Tuple

def create_output_folder(folder_name: str) -> None:
    """Create new folder"""
    os.makedirs(folder_name, exist_ok=True)

def read_csv_file(file_path: str) -> pd.DataFrame:
    """Read data from csv file and returns dataframe"""
    return pd.read_csv(file_path, header=None, names=['Date', 'Value'])

def split_dataframes(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """splits dataframe into two separate dataframes for dates and values"""
    dates_df = pd.DataFrame({'Date': data['Date']})
    values_df = pd.DataFrame({'Value': data['Value']})
    return dates_df, values_df

def save_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """saves dataframe in csv file """
    dataframe.to_csv(file_path, index=False)

if __name__ == "__main__":
    output_folder = 'script1_files'
    create_output_folder(output_folder)

    file_path = 'csv_files/data.csv'
    data = read_csv_file(file_path)

    dates_df, values_df = split_dataframes(data)

    dates_file_path = os.path.join(output_folder, 'X.csv')
    values_file_path = os.path.join(output_folder, 'Y.csv')

    save_to_csv(dates_df, dates_file_path)
    save_to_csv(values_df, values_file_path)