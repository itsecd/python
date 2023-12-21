
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


def split_dataset(
    img_list: list,
    labels: list,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> tuple:
    """
    Split the dataset into training, validation, and test sets.

    Parameters:
    - img_list: List of image file paths.
    - labels: List of corresponding labels.
    - train_size: Percentage of data for training.
    - val_size: Percentage of data for validation.
    - test_size: Percentage of data for testing.

    Returns:
    - tuple: img_train, labels_train, img_val, labels_val, img_test, labels_test
    """
    total_size = len(img_list)

    print(f"Total dataset size: {total_size}")

    train_size = int(total_size * train_size)
    val_size = int(total_size * val_size)
    test_size = int(total_size * test_size)

    print(f"Training dataset size: {train_size}")
    print(f"Validation dataset size: {val_size}")
    print(f"Test dataset size: {test_size}")

    if train_size <= 0:
        raise ValueError("Not enough samples for training.")

    combined = list(zip(img_list, labels))
    random.seed(42)
    random.shuffle(combined)
    img_list[:], labels[:] = zip(*combined)

    img_val, labels_val = img_list[:val_size], labels[:val_size]
    img_test, labels_test = img_list[val_size:val_size +
                                     test_size], labels[val_size:val_size + test_size]
    img_train, labels_train = img_list[val_size + test_size:val_size + test_size +
                                       train_size], labels[val_size + test_size:val_size + test_size + train_size]

    return img_train, labels_train, img_val, labels_val, img_test, labels_test


def load_dataset(
    csv_path: str,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> tuple:
    """
    Load dataset from a CSV file and split it into training, validation, and test sets.

    Parameters:
    - csv_path: Path to the CSV file containing image annotations.
    - train_size: Percentage of data for training.
    - val_size: Percentage of data for validation.
    - test_size: Percentage of data for testing.

    Returns:
    - tuple: img_train, labels_train, img_val, labels_val, img_test, labels_test
    """
    try:
        dframe = pd.read_csv(
            csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"]
        )
        img_list = dframe["Absolute path"].tolist()
        labels = dframe["Class"].tolist()

        if not img_list or not labels:
            raise ValueError("Empty dataset: No images or labels found.")

        img_list, labels = list(img_list), list(labels)

        combined = list(zip(img_list, labels))
        random.seed(42)
        random.shuffle(combined)
        img_list[:], labels[:] = zip(*combined)

        img_train, labels_train, img_val, labels_val, img_test, labels_test = split_dataset(
            img_list, labels, train_size=train_size, val_size=val_size, test_size=test_size
        )

        return img_train, labels_train, img_val, labels_val, img_test, labels_test
    except FileNotFoundError:
        print(f"Error: File not found at path '{csv_path}'")
        return [], [], [], [], [], []
    except pd.errors.EmptyDataError:
        print(f"Error: Empty file at path '{csv_path}'")
        return [], [], [], [], [], []
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return [], [], [], [], [], []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], [], [], [], [], []


def calculate_accuracy(predictions: list, true_labels: list) -> float:
    """
    Calculate accuracy given predicted and true labels.

    Parameters:
    - predictions: Predicted labels.
    - true_labels: True labels.

    Returns:
    - float: Accuracy.
    """
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    total = len(predictions)
    return correct / total


