import os
import pandas as pd
from typing import Tuple
from datetime import datetime
from file_manipulation import create_output_folder, read_csv_file, convert_to_datetime


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
    group.to_csv(file_path, index=False)


def read_data_from_years(date: datetime, folder_path: str) -> str:
    """Reading data from the Years folder"""
    data = None

    if os.path.exists(folder_path):
        matching_file = None
        date_to_find = date.date()
        
        for file in os.listdir(folder_path):
            file_date_parts = file.split('_')
            if len(file_date_parts) == 2:
                start_date_str, end_date_str = file_date_parts[0], file_date_parts[1].split('.')[0]
                start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
                end_date = datetime.strptime(end_date_str, "%Y%m%d").date()
                if start_date <= date_to_find <= end_date:
                    matching_file = file
                    break

        if matching_file:
            file_path = os.path.join(folder_path, matching_file)
            df = pd.read_csv(file_path)

            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            filtered_data = df[df['Date'].dt.date == date_to_find]

            if not filtered_data.empty:
                value = filtered_data.iloc[0]['Value']
                if value != "Page not found":
                    data = value

    return data


if __name__ == "__main__":
    output_folder = 'script2_files'
    create_output_folder(output_folder)

    file_path = 'dataset/data.csv'
    data = read_csv_file(file_path)
    data = convert_to_datetime(data)

    for year, group in group_data_by_year(data):
        save_group_to_csv(output_folder, year, group)