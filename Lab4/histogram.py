import logging
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from df_functions import class_filter
from preprocess import preprocess_text


logging.basicConfig(level=logging.INFO)


def build_histogram(df: pd.DataFrame, class_label: int) -> None:
    """This function takes dataframe and show it's graph by class"""
    try:
        filtered_df = preprocess_text(class_filter(df, class_label))
        reviews = ''
        for review in filtered_df['Текст отзыва']:
            reviews += ' '
            reviews += review
        splitted = reviews.split()

        letter_counts = Counter(splitted)
        frequencies = letter_counts.values()
        names = letter_counts.keys()

        plt.bar(names, frequencies)
        plt.xlabel('Слово')
        plt.xticks(rotation=45)
        plt.ylabel('Количество слов')
        plt.title('График частоты слов по рейтингу')
        plt.tight_layout()
        plt.show(block=True)  
    except Exception as exc:
        logging.error(f"Can not build histogram: {exc}\n{exc.args}\n")