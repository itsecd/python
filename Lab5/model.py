from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label = os.path.basename(os.path.dirname(img_path))
        label = 0 if label == "brown_bear" else 1
        return img_transformed, label

def load_dataset(csv_path: str) -> list:
    dframe = pd.read_csv(
        csv_path, delimiter=",", names=["Absolute Path", "Relative Path", "Class"]
    )
    img_list = dframe["Absolute Path"].tolist()
    random.shuffle(img_list)
    return img_list

def split_data(img_list) -> list:
    train_list = img_list[0 : int(len(img_list) * 0.8)]
    test_list = img_list[int(len(img_list) * 0.8) : int(len(img_list) * 0.9)]
    val_list = img_list[int(len(img_list) * 0.9) : int(len(img_list))]
    return train_list, test_list, val_list

def transform_data(train_list, test_list, val_list) -> Dataset:
    fixed_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_data = Dataset(train_list, transform=fixed_transforms)
    test_data = Dataset(test_list, transform=fixed_transforms)
    val_data = Dataset(val_list, transform=fixed_transforms)
    return train_data, test_data, val_data

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(3 * 3 * 16, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def show_results(epochs, acc, loss, v_acc, v_loss) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(range(epochs), acc, color="orange", label="Train accuracy")
    ax2.plot(range(epochs), loss, color="orange", label="Train loss")
    ax1.plot(range(epochs), v_acc, color="steelblue", label="Validation accuracy")
    ax2.plot(range(epochs), v_loss, color="steelblue", label="Validation loss")
    ax1.legend()
    ax2.legend()
    plt.show()


