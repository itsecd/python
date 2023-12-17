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
    fixed_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_data = dataset(training_list, transform=fixed_transforms)
    test_data = dataset(testing_list, transform=fixed_transforms)
    val_data = dataset(validation_list, transform=fixed_transforms)
    return train_data, test_data, val_data


def main(csv_dataset) -> None:
    img_list = upload_dataset(csv_dataset)
    training_list, testing_list, validation_list = divide_data(img_list)



if __name__ == "__main__":
    result_list = main("Lab2\set\dataset.csv")
