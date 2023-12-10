import logging
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from df_functions import group_by_class, make_dataframe
from csv_functions import save_csv


logging.basicConfig(level=logging.INFO)
russian_stopwords = stopwords.words("russian")


def build_histogram(df: pd.DataFrame, class_label: int) -> None:
    try: 
        df.loc[:, 'Рейтинг'] = pd.to_numeric(df['Рейтинг'])
        plt.plot(df['Рейтинг'], df[''])

    except Exception as exc:
        logging.error(f"Can not build histogram: {exc}\n{exc.args}\n")


def preprocess_text(df: pd.DataFrame, file_path: str) -> None
    reviews = []
    for review in df['Текст отзыва']:
        tokens = Mystem().lemmatize(review.lower())
        tokens = [token for token in tokens if token not in russian_stopwords\
                  and token != " " \
                  and token.strip() not in punctuation]

        text = " ".join(tokens)
        reviews.append(text)
    df['Текст отзыва'] = reviews
    save_csv(df, file_path)



