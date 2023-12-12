import pandas as pd
import csv


def make_data_frame(path_csv='1.csv') -> pd.DataFrame:
    """
    creates a dataframe based on csv-file
    """
    texts = []
    stars = []
    cnt_words = []
    with open(path_csv, 'r', newline='') as csvfile:
        for rev in csv.reader(csvfile, delimiter=','):
            stars.append(rev[2])
            with open(rev[0], 'r', encoding='utf-8') as f:
                text = f.readline()
                texts.append(text)
                cnt_words.append(len(text.split()))

    df = pd.DataFrame({'review': texts, 'count words': cnt_words, 'stars': stars}).dropna()
    return df


def filter_by_cnt_word(data: pd.DataFrame, cnt: int) -> pd.DataFrame:
    """
    Filter dataframe be count of words in rev
    """
    return data[data['count words'] <= cnt]

def filter_by_class(data: pd.DataFrame, star: int) -> pd.DataFrame:
    """
    Filter dataframe by star of rev
    """
    return data[data['stars'] == str(star)]


def group(data : pd.DataFrame) -> pd.DataFrame:
    """
    Grouping dataframe by count words
    """
    return data.groupby('stars').agg({'count words': ['max', 'min', 'mean']})


if __name__ == '__main__':
    print(group(make_data_frame()))
