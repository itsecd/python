from string import punctuation
from nltk.corpus import stopwords
from pymystem3 import Mystem
import pandas as pd
from csv_functions import save_csv


russian_stopwords = stopwords.words("russian")


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    reviews = []
    for review in df['Текст отзыва']:
        tokens = Mystem().lemmatize(review.lower())
        tokens = [token for token in tokens if token not in russian_stopwords\
                    and token != " " and token.strip() not in punctuation]

        text = " ".join(tokens)
        reviews.append(text)
    df['Текст отзыва'] = reviews
    return df