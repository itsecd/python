import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import sys
from random import shuffle

import nltk
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

sys.path.append('D:\Study\Applied Programming (Python)\Applied-Programming\Lab4')
from df_works import df_build, lemmatize_text


def binarize(data: pd.DataFrame, rating: int) -> pd.DataFrame:
    data.dropna(inplace=True)
    change_labels = lambda x: 1 if x==rating else 0
    data['Label'] = data['Rating'].apply(change_labels)
    return data

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(10000, 100)
        self.linear2 = nn.Linear(100, 10)
        self.linear3 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

data = df_build('D:\Study\Applied Programming (Python)\Applied-Programming\csv\dataset.csv')

remove_non_alphabets =lambda x: re.sub(r'[^a-zA-Z]',' ',x)
data = lemmatize_text(data)

max_words = 1000
cv = CountVectorizer(max_features=max_words, stop_words=stopwords.words('russian'))
sparse_matrix = cv.fit_transform(data['Review text']).toarray()
print(cv.get_feature_names_out())

