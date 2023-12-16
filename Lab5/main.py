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
from typing import Tuple


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


def transform_dataset(train_list,test_list,valid_list) -> Tuple[dataset,dataset,dataset]:
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    train_data=dataset(train_list,transform=transform)
    test_data=dataset(test_list,transform=transform)
    valid_data=dataset(valid_list,transform=transform)
    return train_data,test_data,valid_data


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.layer1=nn.Sequential(nn.Conv2d(3,16,kernel_size=3,padding=0,stride=2),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2))
        
        self.layer2=nn.Sequential(nn.Conv2d(16,32,kernel_size=3,padding=0,stride=2),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))

        self.layer3=nn.Sequential(nn.Conv2d(32,64,kernel_size=3,padding=0,stride=2),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2))

        self.fc1=nn.Linear(3*3*64,10)
        self.dropout=nn.Dropout(0.5)
        self.fc2=nn.Linear(10,2)
        self.relu=nn.ReLU()

    def forward(self,x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.view(output.size(0),-1)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        return output


def show_results(epochs,acc,loss) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(range(epochs), acc, color="green", label="Train accuracy")
    ax2.plot(range(epochs), loss, color="green", label="Train loss")
    ax1.legend()
    ax2.legend()
    plt.show()


def train_loop(epochs, batch_size, lear, val_data, train_data, test_data) -> list:
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

    epochs = epochs
    accuracy_values = []
    loss_values = []
    val_accuracy_values = []
    val_loss_values = []
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data, batch_size=batch_size, shuffle=False
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
            "Epoch : {}, train accuracy : {}, train loss : {}".format(
                epoch + 1, epoch_accuracy, epoch_loss
            )
        )
        accuracy_values.append(epoch_accuracy.item())
        loss_values.append(epoch_loss.item())

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

            print(
                "Epoch : {}, val_accuracy : {}, val_loss : {}".format(
                    epoch + 1, epoch_val_accuracy, epoch_val_loss
                )
            )
            val_accuracy_values.append(epoch_val_accuracy.item())
            val_loss_values.append(epoch_val_loss.item())
    show_results(epochs, accuracy_values, loss_values)
    show_results(epochs, val_accuracy_values, val_loss_values)

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


def save_result(rose_probs, csv_path) -> None:
    idx = list(i for i in range(len(rose_probs)))
    prob = list(map(lambda x: x[1], rose_probs))
    submission = pd.DataFrame({"id": idx, "label": prob})
    submission.to_csv(csv_path, index=False)


def main(csv_dataset, epochs, batch_size, lear, result, model_path) -> None:
    """
    """
    img_list = load_dataset(csv_dataset)
    train_list, test_list, valid_list = split_dataset(img_list)
    train_data, test_data, valid_data = transform_dataset(train_list, test_list, valid_list)
    rose_probs, model = train_loop(
        epochs, batch_size, lear, train_data, test_data, valid_data
    )
    save_result(rose_probs, result)
    class_ = {0: "tiger", 1: "leopard"}
    fig, axes = plt.subplots(1, 5, figsize=(20, 12), facecolor="w")
    submission = pd.read_csv(result)
    for ax in axes.ravel():
        i = random.choice(submission["id"].values)
        label = submission.loc[submission["id"] == i, "label"].values[0]
        if label > 0.5:
            label = 1
        else:
            label = 0

        img_path = train_list[i]
        img = Image.open(img_path)

        ax.set_title(class_[label])
        ax.imshow(img)
    plt.show()
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train_list = main(
        "Lab2/file.csv", 1, 100, 0.001, "Lab5/result.csv", "Lab5/result.pt"
    )