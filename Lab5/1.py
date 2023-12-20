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
import nltk
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

sys.path.insert(1, "D:/AppProgPython/appprog/Lab4")
from df_functions import make_dataframe
from preprocess import preprocess_text

def load_data(file_path: str) -> torch.Tensor:


    data = make_dataframe(file_path)
    data = preprocess_text(data)
    cv = CountVectorizer(max_features=10000, stop_words="russian")
    sparse_matrix = cv.fit_transform(data["Текст отзыва"]).toarray()
    sparse_matrix.shape

load_data('D:/AppProgPython/appprog/csv/3.csv')