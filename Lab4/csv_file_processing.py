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
    print(statistics)
    return data


def filter_by_mean_deviation(dataframe, deviation_value):
    filtered_dataframe = dataframe[dataframe['Mean_Deviation'] >= deviation_value]
    return filtered_dataframe


def filter_by_date_range(dataframe, start_date, end_date):
    filtered_dataframe = dataframe[(dataframe['Date'] >= start_date) & (dataframe['Date'] <= end_date)]
    return filtered_dataframe


def group_by_month_and_mean(dataframe):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    grouped_by_month = dataframe.groupby(pd.Grouper(key='Date', freq='M')).mean()

    return grouped_by_month
