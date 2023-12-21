import logging
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from pymystem3 import Mystem


logging.basicConfig(level=logging.INFO)


def df_build(csv_path: str) -> pd.DataFrame:
    """Returns a pandas dataframe based on .csv file"""
    df = pd.read_csv(csv_path, delimiter=',', names=['Abs path', 'Rel path', 'Rating'])
    df = df.drop('Rel path', axis=1)

    abs_path = df['Abs path']
    review_list = []
    for path in abs_path:
        review = open(path, 'r', encoding='utf-8').read()
        review_list.append(review)
    df['Review text'] = review_list

    word_count = df['Review text'].str.count(' ') + 1
    df['Word count'] = word_count
    return df


def stats_df(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a pandas dataframe with word count and rating stats from given dataframe"""
    stats = df[['Rating', 'Word count']].describe()
    return stats


def rating_filter(df: pd.DataFrame, rating: int) -> pd.DataFrame:
    """Returns a dataframe filtered by reviews' rating"""
    df['Rating'] = pd.to_numeric(df['Rating'], errors = 'coerce')
    filtered = df[df['Rating'] == rating].reset_index()
    return filtered


def word_count_filter(df: pd.DataFrame, word_count: int) -> pd.DataFrame:
    """Returns a dataframe filtered by word count"""
    filtered = df[df['Word count'] <= word_count].reset_index()
    return filtered


def rating_group(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe grouped by reviews' rating"""
    grouped_df = df.groupby('Rating').agg({"Word count": ["min", "max", "mean"]})
    return grouped_df


russian_stopwords = stopwords.words("russian")
def lemmatize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe with lemmatized text"""
    reviews = []
    for review in df['Review text']:
        tokens = Mystem().lemmatize(review.lower())
        tokens = [token for token in tokens if token not in russian_stopwords and token != " " and token.strip() not in punctuation]

        text = " ".join(tokens)
        reviews.append(text)
    df['Review text'] = reviews
    return df