import argparse
import pandas as pd
import csv
import logging
import os


def make_data_frame(path_csv='1.csv') -> pd.DataFrame:
    """
    creates a dataframe based on csv-file
    """
    texts = []
    stars = []
    cnt_words = []
    try:
        with open(path_csv, 'r', newline='') as csvfile:
            for rev in csv.reader(csvfile, delimiter=','):
                stars.append(rev[2])
                with open(rev[0], 'r', encoding='utf-8') as f:
                    text = f.readline()
                    texts.append(text)
                    cnt_words.append(len(text.split()))
    except:
        logging.error('error in make_data_frame')
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


def group(data: pd.DataFrame) -> pd.DataFrame:
    """
    Grouping dataframe by count words
    """
    return data.groupby('stars').agg({'count words': ['max', 'min', 'mean']})


def get_statistic(data: pd.DataFrame) -> pd.DataFrame:
    """
    get statistic about numeric cols
    """
    try:
        return data[['count words']].describe()
    except:
        logging.error('error in get_statistic')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=os.path.join('py_log.log'), filemode='w')
    parser = argparse.ArgumentParser(description="Input csv path, label of class")
    parser.add_argument("-c", "--csv", help="Input csv path", type=str)
    args = parser.parse_args()
    make_data_frame(args.csv)
