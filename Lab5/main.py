from matplotlib import pyplot as plt
from typing import Tuple, Any
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import logging
from torchvision import transforms
import sys

logging.basicConfig(level=logging.DEBUG)


def is_balanced(df: pd.DataFrame) -> bool:
    """Checking for the balance of the DataFrame by class"""
    class_stats = df['class'].value_counts()
    return class_stats.min() / class_stats.max() >= 0.98

def img_upload(path_csv: str) -> pd.DataFrame:
    """return pd.DataFrame of list of dataset"""
    data = pd.read_csv(path_csv,  usecols=[0, 2])
    data.columns = ['absolute_path', 'class']
    return data

def split_data(df: pd.DataFrame) -> list:
    """Divides the Data Frame into a training, test, and validation sample"""
    splited_by_class, train_list, test_list, val_list = list(), list(), list(), list()
    names = set(df["class"].to_list())
    i = 0
    for name in names:
        splited_by_class.append(df.loc[df['class'] == name, "absolute_path"])
        train_list.append(splited_by_class[i][0 : int(len(splited_by_class[i]) * 0.8)])
        test_list.append(splited_by_class[i][int(len(splited_by_class[i]) * 0.8) : int(len(splited_by_class[i]) * 0.9)])
        val_list.append(splited_by_class[i][int(len(splited_by_class[i]) * 0.9) : int(len(splited_by_class[i]))])
        i+=1
    return train_list, test_list, val_list

class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(576, 10)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = torch.nn.Flatten()(output)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        return torch.nn.Sigmoid()(output)
    
class CustomDataset(Dataset):
  """Class to store images"""
  def __init__(self, annotation_file: str, transform: Any = None, target_transform: Any = None) -> None:
    self.path_to_annotation = annotation_file
    self.dataset_info = pd.read_csv(annotation_file)
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self) -> int:
    return len(self.dataset_info)

  def __getitem__(self, index: int) -> Tuple[torch.tensor, int]:
    path_to_image = self.dataset_info.iloc[index, 0]
    image = cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)
    label = self.dataset_info.iloc[index, 1]

    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)

    return image, label


def pipeline(train_list, test_list, val_list) -> CustomDataset :
    """Pipeline of data preprocessing"""
    custom_transforms = transforms.Compose(
       [
          torchvision.transforms.ToTensor(),
          transforms.Resize((224, 224)),
          torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
          ]
    )
    train_data = CustomDataset(train_list, transform=custom_transforms)
    test_data = CustomDataset(test_list, transform=custom_transforms)
    val_data = CustomDataset(val_list, transform=custom_transforms)
    return train_data, test_data, val_data