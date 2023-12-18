import pandas as pd
import torch
import torch.nn as nn
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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out

def create_sequence_data(data, seq_length):
    sequences = []
    labels = []

    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length:i+seq_length+1]
        sequences.append(seq)
        labels.append(label)

    sequences_tensor = torch.stack(sequences)
    labels_tensor = torch.squeeze(torch.stack(labels))

    return sequences_tensor, labels_tensor

file_path = 'Lab5/dataset.csv'

data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sequences, labels = create_sequence_data(data, seq_length=3)

print("Полученные последовательности:")
print(sequences)
print("Соответствующие метки:")
print(labels)