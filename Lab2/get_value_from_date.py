import pandas as pd
from datetime import datetime, timedelta
import os
import csv
from typing import Tuple, Union


def read_data_from_dataset(date: datetime, file_path: str) -> str:
    """Reading data from the dataset folder"""

    data = None
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
        date_to_compare = date.date()

        if date_to_compare in df['Date'].dt.date.values:
            value = df.loc[df['Date'].dt.date == date_to_compare, 'Value'].values[0]
            if value != "Page not found":
                data = value

    return data


def read_data_from_script1_files(date: datetime, dates_file_path: str, values_file_path: str) -> str:
    """Reading data from the script1_files folder"""

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


def read_data_from_script2_files(date: datetime, folder_path: str) -> str:
    """Reading data from the script2_files folder"""
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


def read_data_from_script3_files(date: datetime, folder_path: str) -> str:
    """Reading data from the script3_files folder"""
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


def is_one_year_difference(start_date_str: str, end_date_str: str) -> bool:
    """Check date difference in file name(must be one year)"""
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    
    return abs((end_date - start_date).days) in (365, 366)


def is_one_week_difference(start_date_str: str, end_date_str: str) -> bool:
    """Check date difference in file name(must be one week)"""
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    
    return abs((end_date - start_date).days) == 7


def get_data_for_date(date: datetime) -> str:
    """Getting data for the specified date from files in folders"""
    data = None

    main_directory = os.getcwd()
    
    for folder_path, _, files in os.walk(main_directory):
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            date_range = file_name.split(".")[0].split("_")
            if len(date_range) != 2:
                continue

            start_date_str, end_date_str = date_range[0], date_range[1]

            try:
                if is_one_week_difference(start_date_str, end_date_str):
                    data = read_data_from_script3_files(date, folder_path)
                    break
                elif file_name.endswith("X.csv"):
                    data = read_data_from_script1_files(date, folder_path)
                    break
                elif file_name.endswith("Y.csv"):
                    continue
                elif is_one_year_difference(start_date_str, end_date_str):
                    data = read_data_from_script2_files(date, folder_path)
                    break
                elif file_name == "data.csv":
                    data = read_data_from_dataset(date, folder_path)
                    break
            except ValueError:
                continue

        if data is not None:
            break

    return data


if __name__ == "__main__":
    date_to_find = datetime(2023, 9, 5)

    data_for_date = get_data_for_date(date_to_find)
    print(f"Value for: {date_to_find}: {data_for_date}")
