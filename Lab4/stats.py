import argparse
import logging
import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt


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
    


def show_figure(df: pd.DataFrame) -> None:
    """the function takes dataframe and show it's graph"""
    try:
        df.loc[:, 'Дата'] = pd.to_datetime(df['Дата'])
        plt.figure(figsize=(10, 6))
        plt.plot(df['Дата'], df['Курс'])
        plt.title('График изменения курса за весь период')
        plt.xlabel('Дата')
        plt.ylabel('Курс')
        plt.show()
    except Exception as ex:
        logging.error(f"can't show figure: {ex}\n{ex.args}\n")


def show_figure_month(df: pd.DataFrame,
                      target_month: str
                      ) -> None:
    """the function takes dataframe and show it's graph by month"""
    try:
        df['Дата'] = pd.to_datetime(df['Дата'])
        df['Месяц'] = df['Дата'].dt.to_period('M')

        monthly_data = df[df['Месяц'] == target_month]
        median_value = monthly_data['Курс'].median()
        mean_value = monthly_data['Курс'].mean()
    except Exception as ex:
        logging.error(f"can't get monthly data: {ex}\n{ex.args}\n")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_data['Дата'], monthly_data['Курс'], label='Курс')
        plt.axhline(median_value, color='red', linestyle='--', label='Медиана')
        plt.axhline(mean_value, color='green', linestyle='--', label='Среднее значение')

        plt.title(f'График изменения курса за {target_month}')
        plt.xlabel('Дата')
        plt.ylabel('Курс')
        plt.legend()
        plt.show()
    except Exception as ex:
        logging.error(f"can't show figure: {ex}\n{ex.args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='get statistics on a csv file')
    parser.add_argument('--path_file',
                        type=str, default="dataset/dataset.csv",
                        help='the path to the csv file with the data'
                        )
    parser.add_argument('--new_path',
                        type=str, default="dataset/stat.csv",
                        help='new path of modified csv'
                        )
    parser.add_argument('--deviation_value',
                        type=float, default=20.0,
                        help='deviation value for filter'
                        )
    parser.add_argument('--start_date',
                        type=datetime, default=datetime(2023,9,1),
                        help='start date for filter'
                        )
    parser.add_argument('--end_date',
                        type=datetime, default=datetime(2023,10,1),
                        help='end date for filter'
                        )
    parser.add_argument('--target_month',
                        type=str, default="2023-09",
                        help='target month for group'
                        )
    args = parser.parse_args()
    df = form_dataframe(args.path_file)
    deviation_df = filter_by_deviation(df,args.deviation_value)
    date_df = filter_by_date(df,args.start_date,args.end_date)
    monthly_df, monthly_avg = group_by_month(df)
    show_figure(df)
    show_figure(deviation_df)
    show_figure(date_df)
    show_figure(monthly_df)
    show_figure_month(df,args.target_month)