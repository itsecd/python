from string import punctuation
from nltk.corpus import stopwords
from pymystem3 import Mystem
import pandas as pd
import nltk
nltk.download('stopwords')
russian_stopwords = stopwords.words("russian")


def preprocess_reviews_text(dataframe):
    """This function performs text preprocessing by lemmatizing the text in the dataframe."""
    lemmatized_reviews = []
    for review in dataframe['Текст отзыва']:
        tokens = Mystem().lemmatize(review.lower())
        tokens = [token for token in tokens if token not in russian_stopwords and token.strip() not in punctuation]
        lemmatized_review = " ".join(tokens)
        lemmatized_reviews.append(lemmatized_review)
    dataframe['Processed Text'] = lemmatized_reviews
    return dataframe