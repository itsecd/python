import pandas as pd
import logging


def load_reviews(csv_path):
    try:
        df = pd.read_csv(csv_path, delimiter=",", names=['Absolute Path', 'Relative', 'Class'])
        df.drop('Relative', axis=1, inplace=True)

        reviews = []
        for path in df['Absolute Path']:
            try:
                with open(path, "r", encoding="utf-8") as file:
                    review = file.read()
                    reviews.append(review)
            except Exception as e:
                logging.error(f"Can't read file at {path}: {e}")
                reviews.append(None)

        df['Текст отзыва'] = reviews
        df.dropna(subset=['Текст отзыва'], inplace=True)

        df['Количество слов'] = df['Текст отзыва'].str.split().apply(len)
        return df
    except Exception as e:
        logging.error(f"Can't create dataframe: {e}")


def get_reviews_statistics(df):
    try:
        statistics = df[['Рейтинг', 'Количество слов']].describe()
        return statistics
    except Exception as e:
        logging.error(f"Can't get statistics: {e}")


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