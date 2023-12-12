import argparse
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from df_manip import filter_by_class, make_data_frame


def make_hist(df: pd.DataFrame, star: int) -> list:
    """
    does lemmatic analysis
    """
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    data = filter_by_class(df, star)
    for i in range(len(data)):
        for word in word_tokenize(data['review'].values[i]):
            if ord(word[0]) > 191:
                lemmas.append(lemmatizer.lemmatize(word))
    return lemmas


def draw_hists(df: pd.DataFrame) -> None:
    """
    draw histogramm
    """
    plt.bar(range(1, 6), [len(make_hist(df, i)) for i in range(1, 6)])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input csv path, label of class")
    parser.add_argument("-c", "--csv", help="Input csv path", type=str)
    args = parser.parse_args()
    draw_hists(make_data_frame(args.csv))
