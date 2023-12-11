import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)


def preparing_dataframe(file_path: str) -> pd.DataFrame:
    """Process a CSV file to create a DataFrame with valid data and prepare it for graphs and filters."""

    try:
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
    except Exception as ex:
        logging.error(f"Can't form dataframe: {ex}\n{ex.args}\n")


def filter_by_mean_deviation(dataframe: pd.DataFrame, deviation_value: float) -> pd.DataFrame:
    """Filter DataFrame based on the mean deviation value."""

    try:
        filtered_dataframe = dataframe[dataframe['Mean_Deviation'] >= deviation_value]
        return filtered_dataframe
    except Exception as ex:
        logging.error(f"can't filter by mean deviation: {ex}\n{ex.args}\n")


def filter_by_date_range(dataframe: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Filter DataFrame based on a specified date range."""

    try:
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        filtered_dataframe = dataframe[(dataframe['Date'] >= start_date) & (dataframe['Date'] <= end_date)]
        return filtered_dataframe
    except Exception as ex:
        logging.error(f"can't filter by date range: {ex}\n{ex.args}\n")


def group_by_month(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Group DataFrame by month."""

    try:
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        dataframe['Month'] = dataframe['Date'].dt.to_period('M')
        return dataframe
    except Exception as ex:
        logging.error(f"can't group by month: {ex}\n{ex.args}\n")