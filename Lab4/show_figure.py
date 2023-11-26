import logging
import matplotlib.pyplot as plt
import pandas as pd


logging.basicConfig(level=logging.INFO)


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