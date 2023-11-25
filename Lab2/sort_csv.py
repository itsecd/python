from file_manipulation import create_output_folder,  read_csv_file
from splitting_into_two_files import split_dataframes, generate_and_save_files;
from division_by_week import convert_to_datetime, split_data_by_weeks
from division_by_year import group_data_by_year, save_group_to_csv
import os

def split_into_two(output_folder):
    create_output_folder(output_folder)
    file_path = "Lab2/dataset/data.csv"
    data = read_csv_file(file_path)

    dates_df, values_df = split_dataframes(data)

    generate_and_save_files(output_folder, dates_df, values_df)


def sort_by_week(output_folder):
    create_output_folder(output_folder)

    file_path = 'Lab2/dataset/data.csv'
    data = read_csv_file(file_path)
    data = convert_to_datetime(data)

    for start_date, end_date, week_data in split_data_by_weeks(data):
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        file_name = f"{start_date_str}_{end_date_str}.csv"
        
        file_path = os.path.join(output_folder, file_name)
        week_data.to_csv(file_path, index=False)


def sort_by_year(output_folder):
    create_output_folder(output_folder)

    file_path = 'Lab2/dataset/data.csv'
    data = read_csv_file(file_path)
    data = convert_to_datetime(data)

    for year, group in group_data_by_year(data):
        save_group_to_csv(output_folder, year, group)