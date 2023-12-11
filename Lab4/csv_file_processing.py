import pandas as pd


def preparing_dataframe(file_path):
    data = pd.read_csv(file_path)
    data.columns = ['Date', 'Value']

    invalid_values = ['NaN', None, 'Page not found']  # Список невалидных значений для проверки
    for column in data.columns:
        data = data[~data[column].isin(invalid_values)]
    data.reset_index(drop=True, inplace=True)

    data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
    data = data.dropna(subset=['Value'])
    median_value = data['Value'].median()
    mean_value = data['Value'].mean()
    data['Median_Deviation'] = data['Value'] - median_value
    data['Mean_Deviation'] = data['Value'] - mean_value
    statistics = data[['Value', 'Median_Deviation', 'Mean_Deviation']].describe()

    return statistics


def main(file_path):
    data = preparing_dataframe(file_path)
    return data


file_path = "Lab4/dataset/data.csv"
resulting_dataframe = main(file_path)
print(resulting_dataframe)