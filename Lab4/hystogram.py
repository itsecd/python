import logging
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from df_works import rating_filter, lemmatize_text


logging.basicConfig(level=logging.INFO)


def build_histogram(df: pd.DataFrame, rating: int) -> None:
    """This function takes dataframe and show it's graph by class"""
    try:
        filtered_df = lemmatize_text(rating_filter(df, rating))
        reviews = [f'{review} ' for review in filtered_df['Review text']]
        splitted = str(reviews).split()

        word_counts = Counter(splitted)
        frequencies = word_counts.values()
        names = word_counts.keys()

        plt.bar(names, frequencies)
        plt.xlabel('Word')
        plt.xticks(rotation=90)
        plt.ylabel('Word count')
        plt.title('Word use frequency by rating')
        plt.show(block=True)  
    except Exception as exc:
        logging.error(f"Can not build histogram: {exc}\n{exc.args}\n")