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
        self.linear1 = nn.Linear(1000, 100)
        self.linear2 = nn.Linear(100, 10)
        self.linear3 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
def split_data(review_list) -> (list, list, list):
    """Function splits the list into three sublists
    (train_list, test_list, val_list) in a ratio of 80:10:10"""
    train_list = review_list[0 : int(len(review_list) * 0.8)]
    test_list = review_list[int(len(review_list) * 0.8) : int(len(review_list) * 0.9)]
    val_list = review_list[int(len(review_list) * 0.9) : int(len(review_list))]
    return train_list, test_list, val_list

data = df_build('D:\Study\Applied Programming (Python)\Applied-Programming\csv\dataset.csv')

remove_non_alphabets =lambda x: re.sub(r'[^a-zA-Z]',' ',x)
data = lemmatize_text(data)

max_words = 1000
cv = CountVectorizer(max_features=max_words, stop_words=stopwords.words('russian'))
sparse_matrix = cv.fit_transform(data['Review text']).toarray()
l = len(sparse_matrix)
x_train_list, x_test_list, x_val_list = split_data(sparse_matrix[:l//2])
print(x_train_list)
print(x_test_list)
y_train_list, y_test_list, y_val_list = split_data(sparse_matrix[l//2+1:])
print(y_train_list)
print(y_test_list)

'''model = LogisticRegression()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters() , lr=0.01)
x_train = Variable(torch.from_numpy(x_train_list)).float()
y_train = Variable(torch.from_numpy(y_train_list)).long()
epochs = 20
model.train()
loss_values = []
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    print(y_pred.shape)
    loss = criterion(y_pred, y_train)
    loss_values.append(loss.item())
    pred = torch.max(y_pred, 1)[1].eq(y_train).sum()
    acc = pred * 100.0 / len(x_train)
    print('Epoch: {}, Loss: {}, Accuracy: {}%'.format(epoch+1, loss.item(), acc.numpy()))
    loss.backward()
    optimizer.step()

plt.plot(loss_values)
plt.title('Loss Value vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss'])
plt.show()

x_test = Variable(torch.from_numpy(x_test_list)).float()
y_test = Variable(torch.from_numpy(y_test_list)).long()
model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    loss = criterion(y_pred, y_test)
    pred = torch.max(y_pred, 1)[1].eq(y_test).sum()
    print ("Accuracy : {}%".format(100*pred/len(x_test)))'''