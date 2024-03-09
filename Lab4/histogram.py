import logging
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from make_data_frame import filter_reviews_by_class
from lemmatize import preprocess_reviews_text


logging.basicConfig(level=logging.INFO)


def plot_word_frequency(df: pd.DataFrame, class_label: int) -> None:
    """Эта функция генерирует график частоты слов на основе предоставленной метки класса."""
    try:
        filtered_reviews = preprocess_reviews_text(filter_reviews_by_class(df, class_label))
        all_reviews = ' '.join(filtered_reviews['Текст отзыва']).split()

        word_counts = Counter(all_reviews)
        words, frequencies = zip(*word_counts.most_common(20))

        plt.bar(words, frequencies)
        plt.xlabel('Слово')
        plt.xticks(rotation=45)
        plt.ylabel('Количество слов')
        plt.title('Частота слов в отзывах по рейтингу')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Не удалось построить гистограмму : {e}\n{e.args}\n")