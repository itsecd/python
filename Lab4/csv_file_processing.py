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


def add_deviation_columns(data):
    data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
    data = data.dropna(subset=['Value'])

    median_value = data['Value'].median()
    mean_value = data['Value'].mean()

    data['Median_Deviation'] = data['Value'] - median_value
    data['Mean_Deviation'] = data['Value'] - mean_value

    return data


def main(file_path):
    data = read_data(file_path)
    data = rename_columns(data)
    data = remove_invalid_values(data)
    data = add_deviation_columns(data)

    return data


file_path = "Lab4/dataset/data.csv"
resulting_dataframe = main(file_path)
print(resulting_dataframe)