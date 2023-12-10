import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)


def make_dataframe(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, delimiter=",", names=['Абсолютный путь', 'Относительный путь', 'Рейтинг'])

        abs_path = df['Абсолютный путь']
        reviews = []
        for path in abs_path:
            review = open(path, "r", encoding="utf-8").read()
            reviews.append(review)
        df['Текст отзыва'] = reviews

        invalid_reviews = df[df['Текст отзыва'].isnull()]
        if not invalid_reviews.empty:
            df = df.dropna(subset=['Текст отзыва'])
            
        count_words = df['Текст отзыва'].str.count(" ") + 1
        df['Количество слов'] = count_words
        return df
    except Exception as exc:
        logging.error(f"Can't create dataframe: {exc}\n{exc.args}\n")


def create_stats(df: pd.DataFrame) -> pd.DataFrame:
    statistics = df[['Рейтинг', 'Количество слов']].describe()
    return statistics


def count_filter(df: pd.DataFrame, count: int) -> pd.DataFrame:
    filtered_df = df[df['Количество слов'] <= count].reset_index()
    return filtered_df


def class_filter(df: pd.DataFrame, class_label: int) -> pd.DataFrame:
    filtered_df = df[df['Рейтинг'] == class_label].reset_index()
    return filtered_df


def group_by_class(df: pd.DataFrame) -> pd.DataFrame:
    try:
        grouped_df = df.groupby('Рейтинг').agg({"Количество слов": ["min", "max", "mean"]})
        return grouped_df
    except Exception as exc:
        logging.error(f"Can't group Dataframe: {exc}\n{exc.args}\n")


def build_histogram(df: pd.DataFrame, class_label: int) -> list:
    filtered_df = class_filter(df, class_label)
    