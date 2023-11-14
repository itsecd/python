import pandas as pd
from datetime import datetime
import os
import csv

# Пути к папкам с файлами csv
folder_paths = [
    'csv_files',
    'script1_files',
    'script2_files',
    'script3_files'
]

def read_data_from_folder1(date: datetime) -> str:
    """Reading data from the first folder"""

    data = None
    file_path = 'csv_files/data.csv'
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
        date_to_compare = date.date()

        if date_to_compare in df['Date'].dt.date.values:
            value = df.loc[df['Date'].dt.date == date_to_compare, 'Value'].values[0]
            if value != "Page not found":
                data = value

    return data


def read_data_from_folder2(date: datetime) -> str:
    """Reading data from the second folder"""

    data = None
    dates_file_path = 'script1_files/X.csv'
    values_file_path = 'script1_files/Y.csv'

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

def read_data_from_folder3(date: datetime) -> str:
    """Reading data from the third folder"""

    data = None
    folder_path = 'script2_files'

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

def read_data_from_folder4(date: datetime) -> str:
    """Reading data from the fourth folder"""
    data = None
    folder_path = 'script3_files'

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

def get_data_for_date(date: datetime) -> str:
    """Getting data for the specified date from files in folders"""
    data = None

    for folder_path in folder_paths:
        if folder_path == 'csv_files':
            data = read_data_from_folder1(date)
        elif folder_path == 'script1_files':
            data = read_data_from_folder2(date)
        elif folder_path == 'script2_files':
            data = read_data_from_folder3(date)
        elif folder_path == 'script3_files':
            data = read_data_from_folder4(date)

        if data is not None:
            break

    return data



if __name__ == "__main__":
    # Пример использования функций
    date_to_find = datetime(2023, 10, 5)

    data_for_date = get_data_for_date(date_to_find)
    print(f"Value for: {date_to_find}: {data_for_date}")
