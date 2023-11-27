import os
import pandas as pd
from typing import Tuple
from file_manipulation import create_output_folder, read_csv_file                                                                       
from datetime import datetime

def split_dataframes(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """splits dataframe into two separate dataframes for dates and values"""
    dates_df = pd.DataFrame({'Date': data['Date']})
    values_df = pd.DataFrame({'Value': data['Value']})
    return dates_df, values_df


def save_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """saves dataframe in csv file """
    dataframe.to_csv(file_path, index=False)


def generate_and_save_files(output_folder: str, dates_df: pd.DataFrame, values_df: pd.DataFrame) -> None:
    """generates file paths and saves dataframes to corresponding csv files"""
    dates_file_path = os.path.join(output_folder, 'X.csv')
    values_file_path = os.path.join(output_folder, 'Y.csv')

    save_to_csv(dates_df, dates_file_path)
    save_to_csv(values_df, values_file_path)

def read_data_from_xy(date: datetime, dates_file_path: str, values_file_path: str) -> str:
    """Reading data from the XY folder"""

    data = None

    if os.path.exists(dates_file_path) and os.path.exists(values_file_path):
        dates_df = pd.read_csv(dates_file_path)
        values_df = pd.read_csv(values_file_path)
        
        dates_df['Date'] = pd.to_datetime(dates_df['Date'], format='%Y/%m/%d')
        date_to_compare = date.date()
        dates_filtered = dates_df[dates_df['Date'].dt.date == date_to_compare]

        if not dates_filtered.empty:
            index = dates_filtered.index[0]
            value = values_df.iloc[index]['Value']
            if value != "Page not found":
                data = value

    return data

if __name__ == "__main__":
    output_folder = 'script1_files'
    create_output_folder(output_folder)

    file_path = 'dataset/data.csv'
    data = read_csv_file(file_path)

    dates_df, values_df = split_dataframes(data)

    generate_and_save_files(output_folder, dates_df, values_df)