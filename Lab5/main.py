import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image


def load_data(csv_path : str) -> list:
    """Def for load dataset and output list"""
    dframe = pd.read_csv(csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"])
    list_ = dframe["Absolute path"].to_list()
    return list_

def separation_data(images : list) -> list:
    """Separation data lika a train, test and valid"""
    train_data = images[0:int(len(images) * 0.8)]
    test_data = images[int(len(images) * 0.8) : int(len(images) * 0.9)]
    valid_data = images[int(len(images) * 0.9) : len(images)]
    return train_data, test_data, valid_data