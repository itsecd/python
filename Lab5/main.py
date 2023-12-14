import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import zipfile


def load_dataset(path: str) -> list:
    df=pd.read_csv(path,delimiter=',',names=["Absolute Path","Relative path","Tag"])
    list=df["Absolute path"].tolist()
    return list


def split_dataset(list: list) -> list:
    train_data=list[0: int(len(list)*0.8)]
    test_data=list[int(len(list)*0.8):int(len(list)*0.9)]
    valid_data = list[int(len(list) * 0.9) : int(len(list))]
    return train_data, test_data, valid_data

class dataset(torch.utils.data.Dataset):
    def __init__(self,list,transform=None) -> None:
        self.dataset=list
        self.transform=transform

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self,idx:int) -> Tuple[torch.tensor,int]:
        path=self.dataset[idx]
        img=Image.open(path)
        img=self.transform(img)
        img_label=[]
        for i in range(len(self.dataset)):
            label.append(os.path.basename(os.path.dirname(self.dataset[i])))
        label=img_label[idx]
        if label == "tiger":
            label=0
        else: 
            label=1
        return img,label


