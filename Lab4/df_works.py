import logging
import pandas as pd


logging.basicConfig(level=logging.INFO)


def df_build(csv_path: str) -> pd.DataFrame:
    """Creates a pandas dataframe based on .csv file"""
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

if __name__ == '__main__':
    df = df_build('Lab2/csv/dataset.csv')
    print(df)