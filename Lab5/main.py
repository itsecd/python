import pandas as pd
import torch
from torch.utils.data import random_split, TensorDataset, DataLoader

def load_and_preprocess_data(file_path):

    data = pd.read_csv(file_path)
    data = data[data['Value'] != "Page not found"]

    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    data['Value'] = (data['Value'].astype(float) - data['Value'].astype(float).min()) / (data['Value'].astype(float).max() - data['Value'].astype(float).min())

    return torch.tensor(data['Value'].values, dtype=torch.float32)

def split_data(all_data: torch.Tensor, train_ratio: float = 0.8, val_ratio: float = 0.1) -> (torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset):
  
    train_size = int(train_ratio * len(all_data))
    val_size = int(val_ratio * len(all_data))
    test_size = len(all_data) - train_size - val_size

    train_data, valtest_data = random_split(all_data, [train_size, val_size + test_size])
    val_data, test_data = random_split(valtest_data, [val_size, test_size])

    return train_data, val_data, test_data

file_path = 'Lab5/dataset.csv'
all_data = load_and_preprocess_data(file_path)

train_data, val_data, test_data = split_data(all_data)

print(f"Размер обучающей выборки: {len(train_data)}")
print(f"Размер валидационной выборки: {len(val_data)}")
print(f"Размер тестовой выборки: {len(test_data)}")