from datetime import date
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO)


def form_dataframe(file_path: str) -> pd.DataFrame:
    """the function takes a file_path to the data file and return dataframe with 2 new cols"""
    try:
        df = pd.read_csv(file_path,delimiter=",")
        df.columns = ['Дата', 'Курс']
        df['Курс'] = pd.to_numeric(df['Курс'], errors="coerce")
        
        invalid_values = df[df['Курс'].isnull()]
        if not invalid_values.empty:
            df = df.dropna(subset=['Курс'])
            
        median_value = df['Курс'].median()
        mean_value = df['Курс'].mean()
        df['Отклонение от медианы'] = df['Курс'] - median_value
        df['Отклонение от среднего'] = df['Курс'] - mean_value
        
        statistics = df[['Курс', 'Отклонение от медианы', 'Отклонение от среднего']].describe()
        print(statistics)
        return df
    except Exception as ex:
        logging.error(f"Can't form dataframe: {ex}\n{ex.args}\n")    


def filter_by_deviation(df: pd.DataFrame,
                        deviation_value: float
                        ) -> pd.DataFrame:
    """the function takes dataframe, deviation value and return filtered df by value"""
    mean_value = df['Курс'].mean()
    df['Отклонение от среднего'] = df['Курс'] - mean_value

    filtered_df = df[df['Отклонение от среднего'] >= deviation_value].copy()

    filtered_df.drop(columns=['Отклонение от среднего'], inplace=True)

    return filtered_df


def filter_by_date(df: pd.DataFrame,
                   start_date: date,
                   end_date: date,
                   ) -> pd.DataFrame:
    """the function takes dataframe, start date and end date and return filtered df by dates"""
    df['Дата'] = pd.to_datetime(df['Дата'])
    filtered_df = df[(df['Дата'] >= start_date) & (df['Дата'] <= end_date)]
    return filtered_df


def group_by_month(df: pd.DataFrame) -> (pd.DataFrame,pd.DataFrame):
    """the function takes dataframe and return df grouped by month and monthly average value"""
    try:
        df['Дата'] = pd.to_datetime(df['Дата'])
        df['Месяц'] = df['Дата'].dt.to_period('M')
        
        monthly_avg_df = df.groupby('Месяц')['Курс'].mean().reset_index()
        return (df, monthly_avg_df)
    except Exception as ex:
        logging.error(f"can't group by month: {ex}\n{ex.args}\n")