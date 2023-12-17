import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def prepare_data(file_path, batch_size=64):

    data = pd.read_csv(file_path)

    data['label'] = data['value'].diff().fillna(0)

    features_tensor = torch.tensor(data['value'].values, dtype=torch.float32).view(-1, 1) 
    labels_tensor = torch.tensor(data['label'].values, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(features_tensor, labels_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

