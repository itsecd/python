import logging
import pandas as pd


logging.basicConfig(level=logging.INFO)


def make_dataframe(csv_path: str) -> pd.DataFrame:
    """This function takes a file path to the data file and returns dataframe with 2 new cols"""
    try:
        df = pd.read_csv(csv_path, delimiter=",", names=['Абсолютный путь', 'Относительный путь', 'Рейтинг'])
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


def create_stats(df: pd.DataFrame) -> pd.DataFrame:
    """This function takes dataframe and returns dataframe with statistics of rate and number of words"""
    statistics = df[['Рейтинг', 'Количество слов']].describe()
    return statistics


def count_filter(df: pd.DataFrame, count: int) -> pd.DataFrame:
    """This function takes dataframe and returns filtered dataframe by count of words"""
    filtered_df = df[df['Количество слов'] <= count].reset_index()
    return filtered_df


def class_filter(df: pd.DataFrame, class_label: int) -> pd.DataFrame:
    """This function takes dataframe and returns filtered dataframe by rate of reviews"""
    df['Рейтинг'] = pd.to_numeric(df['Рейтинг'], errors = 'coerce')
    filtered_df = df[df['Рейтинг'] == class_label].reset_index()
    return filtered_df


def group_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """This function takes dataframe and returns grooped dataframe by rate of reviews"""
    try:
        grouped_df = df.groupby('Рейтинг').agg({"Количество слов": ["min", "max", "mean"]})
        return grouped_df
    except Exception as exc:
        logging.error(f"Can't group Dataframe: {exc}\n{exc.args}\n")