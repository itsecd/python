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


