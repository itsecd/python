import os
import pandas as pd
from typing import Tuple
from division_by_week import create_output_folder, read_csv_file, convert_to_datetime


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

if __name__ == "__main__":
    output_folder = 'script2_files'
    create_output_folder(output_folder)

    file_path = 'dataset/data.csv'
    data = read_csv_file(file_path)
    data = convert_to_datetime(data)

    for year, group in group_data_by_year(data):
        save_group_to_csv(output_folder, year, group)