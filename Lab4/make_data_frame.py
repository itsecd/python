import pandas as pd
import logging
from pandas.core.frame import DataFrame

FIELDNAMES = ['absolute_path', 'relative_path', 'class_label']


def get_reviews(annotation_path: str) -> DataFrame:
    try:
        df = pd.read_csv(annotation_path, delimiter=",", names=FIELDNAMES)

        paths = df[FIELDNAMES[0]]
        text =[]
        for path in paths:
            text.append(read_text_file(path))
        df['Текст отзыва'] = text
        count_words = df['Текст отзыва'].str.count(" ") + 1
        df['Количество слов'] = count_words
        process_invalid_values(df)
        return df

    except Exception as exc:
        logging.error(f"Can't create dataframe: {exc}\n{exc.args}\n")


def read_text_file(abs_path):
    with open(abs_path, 'r', encoding='utf-8') as file:
        return file.read()


def process_invalid_values(df):
    invalid_values = df.isna().any()
    df.fillna('Unknown', inplace=True)
    return df


def filter_reviews_by_word_count(df, count):
    try:
        filtered_df = df[df['Количество слов'] <= count].reset_index(drop=True)
        return filtered_df
    except Exception as e:
        logging.error(f"Can't filter by word count: {e}")


def filter_reviews_by_rating(df, class_label):
    try:
        df['Рейтинг'] = pd.to_numeric(df['Рейтинг'], errors='coerce')
        filtered_df = df[df['Рейтинг'] == class_label].reset_index(drop=True)
        return filtered_df
    except Exception as e:
        logging.error(f"Can't filter by rating: {e}")


def group_reviews_by_rating(df):
    try:
        grouped_df = df.groupby('Рейтинг').agg({"Количество слов": ["min", "max", "mean"]})
        return grouped_df
    except Exception as e:
        logging.error(f"Can't group by rating: {e}")


def get_reviews_statistics(df):
    try:
        statistics = df[['Рейтинг', 'Количество слов']].describe()
        return statistics
    except Exception as e:
        logging.error(f"Can't get statistics: {e}")


def group_reviews_by_class(df: DataFrame) -> DataFrame:
    try:
        grouped_df = df.groupby('Класс').agg({"Количество слов": ["min", "max", "mean"]})
        return grouped_df
    except Exception as e:
        logging.error(f"Can't group by rating: {e}")