from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
import random
from PIL import Image
import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms


def load_dataset(csv_path: str) -> list:
    """Function for uploading data from an annotation file to a list"""
    dframe = pd.read_csv(
        csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"]
    )
    img_list = dframe["Absolute path"].tolist()
    random.shuffle(img_list)
    return img_list


def split_data(img_list) -> list:
    """Function splits the list into three sublists
    (train_list, test_list, val_list) in a ratio of 80:10:10"""
    train_list = img_list[0 : int(len(img_list) * 0.8)]
    test_list = img_list[int(len(img_list) * 0.8) : int(len(img_list) * 0.9)]
    val_list = img_list[int(len(img_list) * 0.9) : int(len(img_list))]
    return train_list, test_list, val_list


class dataset(torch.utils.data.Dataset):
    """Class will store uploaded and converted images
    and labels of classes 0 - roses 1 - tulips"""

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
            img_label.append(os.path.basename(os.path.dirname(self.file_list[i])))
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label = img_label[idx]
        if label == "rose":
            label = 0
        elif label == "tulip":
            label = 1
        return img_transformed, label


def transform_data(train_list, test_list, val_list) -> dataset:
    """Pipeline of data preprocessing and the formation of a dataset
    that will directly participate in the training"""
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
    val_data = dataset(val_list, transform=fixed_transforms)
    return train_data, test_data, val_data


class Cnn(nn.Module):
    """Convolutional neural network model"""

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


def show_results(epochs, acc, loss, v_acc, v_loss) -> None:
    """Function creates graphs based on the received learning results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(range(epochs), acc, color="orange", label="Train accuracy")
    ax2.plot(range(epochs), loss, color="orange", label="Train loss")
    ax1.plot(range(epochs), v_acc, color="steelblue", label="Validation accuracy")
    ax2.plot(range(epochs), v_loss, color="steelblue", label="Validation loss")
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
    model = Cnn()
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
            preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
            rose_probs += list(zip(list(fileid), preds_list))
    rose_probs.sort(key=lambda x: int(x[0]))
    return rose_probs, model


def save_result(rose_probs, csv_path) -> None:
    """Function for saving the result in csv"""
    idx = list(i for i in range(len(rose_probs)))
    prob = list(map(lambda x: x[1], rose_probs))
    submission = pd.DataFrame({"id": idx, "label": prob})
    submission.to_csv(csv_path, index=False)


def main(csv_dataset, epochs, batch_size, lear, result, model_path) -> None:
    """Function that goes through the full training cycle of a neural network
    with a given number of epochs, learning rate and shows the result of work in 5 pictures
    """
    img_list = load_dataset(csv_dataset)
    train_list, test_list, val_list = split_data(img_list)
    train_data, test_data, val_data = transform_data(train_list, test_list, val_list)
    rose_probs, model = train_loop(
        epochs, batch_size, lear, val_data, train_data, test_data
    )
    save_result(rose_probs, result)
    class_ = {0: "rose", 1: "tulip"}
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
        "Lab2\csv_files\datasets.csv", 10, 100, 0.001, "result.csv", "Lab5\weight1.pt"
    )
