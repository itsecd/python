
import pandas as pd
import csv
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib as plt
import logging

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


def group(data: pd.DataFrame) -> pd.DataFrame:
    """
    Grouping dataframe by count words
    """
    return data.groupby('stars').agg({'count words': ['max', 'min', 'mean']})


def make_hist(df: pd.DataFrame, star: int) ->list:
    lemmatizer = WordNetLemmatizer()
    lemmas=[]
    data=filter_by_class(star)
    for i in range(0, len(data)):
        words = word_tokenize(data['review'].values[i])
        for word in words:
            if (ord(word[0]) > 191):
                lemma = lemmatizer.lemmatize(word)
                lemmas.append(lemma)
    return lemmas


def get_statistic(data: pd.DataFrame) -> pd.DataFrame:
    try:
        return data[['count words']].describe()
    except:
        logging.error('error in get_statistic')
def draw_hists(df: pd.DataFrame) -> None:
    """
    draw histogramm
    """
    plt.plot(len(make_hist(df,1)), color='blue', label='Blue')
    plt.plot(len(make_hist(df,2)), color='green', label='Green')
    plt.plot(len(make_hist(df,3)), color='blue', label='Red')
    plt.plot(len(make_hist(df,4)), color='blue', label='Red')
    plt.plot(len(make_hist(df,5)), color='blue', label='Red')

    plt.xlabel("intensity")
    plt.ylabel("density")
    plt.title("Histograms")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    print(get_statistic(make_data_frame()))
    #draw_hists(make_data_frame())
