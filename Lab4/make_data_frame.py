import pandas as pd
import logging
from pandas.core.frame import DataFrame

FIELDNAMES = ['absolute_path', 'relative_path', 'class_label']

def get_reviews(annotation_path :str) -> DataFrame:
    try:
        df = pd.read_csv(annotation_path, delimiter=",", names=FIELDNAMES)

        paths = df[FIELDNAMES[0]]
        text =[]
        for path in paths:
            text.append(read_text_file(path))
        df['Текст отзыва'] = text
        count_words = df['Текст отзыва'].str.count(" ") + 1
        df['Количество слов'] = count_words
        process_invalid_values(df)
        return df

    except Exception as exc:
        logging.error(f"Can't create dataframe: {exc}\n{exc.args}\n")


def read_text_file(abs_path):
    with open(abs_path, 'r', encoding='utf-8') as file:
        return file.read()


def process_invalid_values(df):
    invalid_values = df.isna().any()
    df.fillna('Unknown', inplace=True)
    return df