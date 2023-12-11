import pandas as pd


def read_data(file_path):
    data = pd.read_csv(file_path)
    return data


def rename_columns(data):
    data.columns = ['Date', 'Value']
    return data


def remove_invalid_values(data):
    invalid_values = ['NaN', None, 'Page not found']  # Список невалидных значений для проверки

    for column in data.columns:
        data = data[~data[column].isin(invalid_values)]

    data.reset_index(drop=True, inplace=True)
    return data


def main(file_path):
    data = read_data(file_path)
    data = rename_columns(data)
    data = remove_invalid_values(data)

    return data
