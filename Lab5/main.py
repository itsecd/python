import torch
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import os
import zipfile


def upload_dataset(csv_path: str) -> list:
    '''загружает данные из файла аннотации в список'''
    dframe = pd.read_csv(
        csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"]
    )
    img_list = dframe["Absolute path"].tolist()
    random.shuffle(img_list)
    return img_list


def divide_data(img_list) -> list:
    '''разделение загруженного набора данных на обучающую, 
    тестовую и валидационую выборки (в соотношении 80:10:10)'''
    training_list = img_list[0: int(len(img_list) * 0.8)]
    testing_list = img_list[int(len(img_list) * 0.8): int(len(img_list) * 0.9)]
    validation_list = img_list[int(len(img_list) * 0.9): int(len(img_list))]
    return training_list, testing_list, validation_list


class dataset(torch.utils.data.Dataset):
    '''для загрузки наборов изображений'''

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    # dataset length
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    # load an one of images
    def __getitem__(self, idx):
        img_label = []
        for i in range(len(self.file_list)):
            img_label.append(os.path.basename(
                os.path.dirname(self.file_list[i])))
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label = img_label[idx]
        if label == "cat":
            label = 0
        elif label == "dog":
            label = 1
        return img_transformed, label


def image_augumentation(training_list, testing_list, validation_list) -> dataset:
    '''изменение масштаба'''
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_data = dataset(training_list, transform=transform)
    test_data = dataset(testing_list, transform=transform)
    val_data = dataset(validation_list, transform=transform)
    return train_data, test_data, val_data


class Cnn(nn.Module):
    '''класс построения сверточной модели'''

    def __init__(self):
        super(Cnn, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(3 * 3 * 64, 10)
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


def show_results(epochs, acc, loss, val_acc, val_loss) -> None:
    '''графики результатов обучения'''
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(range(epochs), acc, color="green", label="Train")
    ax[1].plot(range(epochs), loss, color="green", label="Train")
    ax[0].plot(range(epochs), val_acc, color="crimson", label="Validation")
    ax[1].plot(range(epochs), val_loss, color="crimson", label="Validation")
    ax[0].set_title('Accuracy')
    ax[1].set_title('Loss')
    ax[0].legend()
    ax[1].legend()
    fig.suptitle('The result of the training')
    plt.show()


def save_csv(cat_probs, csv_path) -> None:
    '''сохраненить результат в csv-файл'''
    id = list(i for i in range(len(cat_probs)))
    label = list(map(lambda x: x[1], cat_probs))
    submission = pd.DataFrame({"id": id, "label": label})
    submission.to_csv(csv_path, index=False)


def main(csv_dataset) -> None:
    img_list = upload_dataset(csv_dataset)
    training_list, testing_list, validation_list = divide_data(img_list)
    train_data, test_data, val_data = image_augumentation(
        training_list, testing_list, validation_list)


if __name__ == "__main__":
    train_list = main("Lab2\set\dataset.csv")
