import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
from typing import Tuple, Any


def load_data(csv_path : str) -> list:
    """Def for load dataset and output list"""
    dframe = pd.read_csv(csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"])
    list_ = dframe["Absolute path"].to_list()
    return list_


def separation_data(images : list) -> Tuple[list, list, list]:
    """Separation data lika a train, test and valid"""
    train_data = images[0:int(len(images) * 0.8)]
    test_data = images[int(len(images) * 0.8) : int(len(images) * 0.9)]
    valid_data = images[int(len(images) * 0.9) : len(images)]
    return train_data, test_data, valid_data


class dataset(torch.utils.data.Dataset):
    def __init__(self, list_, transform:Any=None) -> None:
        self.dataset = list_
        self.transform = transform


    def __len__(self) -> int:
        return len(self.dataset)
    

    def __getitem__(self,index : int) -> Tuple[torch.tensor, int]:
        path_to_image = self.dataset.iloc[index, 0]
        label = self.dataset.iloc[index, 1]
        # image = cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)
        img = Image.open(path_to_image)
        img = self.transform(img)
        return img, label
    

def transform_data(train_list, test_list, valid_list) -> Tuple[dataset, dataset, dataset]:
    """Transform dataset"""
    fixed_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_data = dataset(train_list, transform=fixed_transforms)
    test_data = dataset(test_list, transform=fixed_transforms)
    valid_data = dataset(valid_list, transform=fixed_transforms)
    return train_data, test_data, valid_data



