import pandas as pd
import logging
from pandas.core.frame import DataFrame


def load_reviews(csv_path: str) -> DataFrame:
    """Загружает отзывы из файла CSV и возвращает DataFrame."""
    try:
        df = pd.read_csv(csv_path, delimiter=",", names=['Абсолютный путь', 'Относительный путь', 'Класс'])
        df = df.drop('Относительный путь', axis=1)

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


def get_reviews_statistics(df: DataFrame) -> DataFrame:
    """Вычисляет статистику по количеству слов в отзывах для каждого класса."""
    try:
        statistics = df[['Класс', 'Количество слов']].describe()
        return statistics
    except Exception as e:
        logging.error(f"Can't get statistics: {e}")


def filter_reviews_by_word_count(df: DataFrame, count: int) -> DataFrame:
    """Фильтрует отзывы по количеству слов."""
    try:
        filtered_df = df[df['Количество слов'] <= count].reset_index(drop=True)
        return filtered_df
    except Exception as e:
        logging.error(f"Can't filter by word count: {e}")


def filter_reviews_by_class(df: DataFrame, class_label: str) -> DataFrame:
    """Фильтрует отзывы по заданному классу."""
    try:
        filtered_df = df[df['Класс'] == class_label].reset_index(drop=True)
        return filtered_df
    except Exception as e:
        logging.error(f"Can't filter by rating: {e}")


def group_reviews_by_class(df: DataFrame) -> DataFrame:
    """Группирует отзывы по классу и вычисляет статистику количества слов."""
    try:
        grouped_df = df.groupby('Класс').agg({"Количество слов": ["min", "max", "mean"]})
        return grouped_df
    except Exception as e:
        logging.error(f"Can't group by rating: {e}")