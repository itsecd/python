
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        """
        Simple Convolutional Neural Network (CNN) model.

        Parameters:
        - num_classes: Number of classes for classification.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
        - x: Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class CustomDataset(Dataset):
    def __init__(
        self,
        img_paths: list,
        labels: list,
        transform: transforms.Compose = None,
        label_mapping: dict = None,
    ) -> None:
        """
        Custom dataset class for image classification.

        Parameters:
        - img_paths: List of image file paths.
        - labels: List of corresponding labels.
        - transform: Image transformations.
        - label_mapping: Mapping of label strings to numerical indices.
        """
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.label_mapping = label_mapping

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
        - int: Length of the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get item from the dataset.

        Parameters:
        - idx: Index of the item.

        Returns:
        - tuple: (image, label)
        """
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label_str = self.labels[idx]
        label = self.label_mapping[label_str] if self.label_mapping else int(
            label_str)

        return img, torch.tensor(label)


