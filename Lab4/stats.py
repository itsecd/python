import argparse
import pandas as pd
from datetime import datetime, date


def form_dataframe(file_path: str) -> pd.DataFrame:
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


def filter_by_deviation(df: pd.DataFrame,
                        deviation_value: float
                        ) -> pd.DataFrame:
    mean_value = df['Курс'].mean()
    df['Отклонение от среднего'] = df['Курс'] - mean_value

    filtered_df = df[df['Отклонение от среднего'] >= deviation_value]

    filtered_df.drop(columns=['Отклонение от среднего'], inplace=True)

    return filtered_df


def filter_by_date(df: pd.DataFrame,
                   start_date: date,
                   end_date: date,
                   ) -> pd.DataFrame:
    df['Дата'] = pd.to_datetime(df['Дата'])
    filtered_df = df[(df['Дата'] >= start_date) & (df['Дата'] <= end_date)]
    return filtered_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='get statistics on a csv file')
    parser.add_argument('--path_file',
                        type=str, default="dataset/dataset.csv",
                        help='the path to the csv file with the data'
                        )
    parser.add_argument('--new_path',
                        type=str, default="stat.csv",
                        help='new path of modified csv'
                        )
    args = parser.parse_args()
    df = form_dataframe(args.path_file)
    df.to_csv(args.new_path)
    