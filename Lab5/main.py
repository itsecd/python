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
        if label == "rose":
            label = 0
        elif label == "tulip":
            label = 1
        return img, label
    

def transform_data(train_list, test_list, valid_list) -> Tuple[dataset, dataset, dataset]:
    """Transform dataset"""
    custom_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_data = dataset(train_list, transform=custom_transforms)
    test_data = dataset(test_list, transform=custom_transforms)
    valid_data = dataset(valid_list, transform=custom_transforms)
    return train_data, test_data, valid_data


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(576,10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10,2)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.view(output.size(0),-1)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        return output
    


def train_loop(epochs, batch_size, lear, train_data, test_data, valid_data) -> Tuple[list, CNN]:
    """Function is intended for creating and training a neural network model,
    as well as graphing and analyzing the results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)
    if device == "cuda":
        torch.cuda.manual_seed_all(1234)
    model = CNN()
    model.train()

    optimizer = optim.Adam(params=model.parameters(), lr=lear)
    criterion = nn.CrossEntropyLoss()

    accuracy_values = []
    loss_values = []
    valid_accuracy_values = []
    valid_loss_values = []
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data, batch_size=batch_size, shuffle=False
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True
    )
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        print(
           f"Epoch : {epoch + 1}, train accuracy : {epoch_accuracy}, train loss : {epoch_loss}"
        )
        accuracy_values.append(epoch_accuracy.item())
        loss_values.append(epoch_loss.item())

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

            print(
                f"Epoch : {epoch + 1}, val_accuracy : {epoch_val_accuracy}, val_loss : {epoch_val_loss}"
            )
            valid_accuracy_values.append(epoch_val_accuracy.item())
            valid_loss_values.append(epoch_val_loss.item())
    show_results(epochs, accuracy_values, loss_values)
    show_results(epochs, valid_accuracy_values, valid_loss_values)

    rose_probs = []
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=100, shuffle=False
    )
    with torch.no_grad():
        for data, fileid in test_loader:
            data = data.to(device)
            preds = model(data)
            preds_list = functional.softmax(preds, dim=1)[:, 1].tolist()
            rose_probs += list(zip(list(fileid), preds_list))
    rose_probs.sort(key=lambda x: int(x[0]))
    return rose_probs, model


def show_results(epochs, acc, loss, v_acc, v_loss) -> None:
    """Creates graphs based on the learning results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(range(epochs), acc, color="green", label="Train accuracy")
    ax2.plot(range(epochs), loss, color="green", label="Train loss")
    ax1.plot(range(epochs), v_acc, color="blue", label="Validation accuracy")
    ax2.plot(range(epochs), v_loss, color="blue", label="Validation loss")
    ax1.legend()
    ax2.legend()
    plt.show()
