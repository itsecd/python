from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from csv_functions import save_csv


russian_stopwords = stopwords.words("russian")


def preprocess_text(df: pd.DataFrame, file_path: str) -> None:
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

