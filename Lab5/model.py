from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
import glob
import torch
import random
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

if __name__ == "__main__":
    dframe = pd.read_csv(
        "Lab2\csv_files\dataset.csv",
        delimiter=",",
        names=["Absolute path", "Relative path", "Class"],
    )
    img_list = dframe["Relative path"].tolist()
    random.shuffle(img_list)
    train_list = img_list[0 : int(len(img_list) * 0.8)]
    test_list = img_list[int(len(img_list) * 0.8) : int(len(img_list) * 0.9)]
    val_list = img_list[int(len(img_list) * 0.9) : int(len(img_list))]

    random_idx = np.random.randint(1, len(img_list), size=10)
    fig = plt.figure()
    i = 1
    for idx in random_idx:
        ax = fig.add_subplot(2, 5, i)
        img = Image.open(img_list[idx])
        plt.imshow(img)
        i += 1
        plt.axis("off")
        plt.show()
