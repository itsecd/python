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


def create_sequence_data(data, seq_length):
    sequences = []
    labels = []

    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append(seq)
        labels.append(label)

    sequences_tensor = torch.stack(sequences).unsqueeze(-1)
    labels_tensor = torch.stack(labels)

    return sequences_tensor, labels_tensor


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    

if __name__ == "__main__":
    file_path = 'Lab5/dataset.csv'
    data = load_and_preprocess_data(file_path)

    train_data, val_data, test_data = split_data(data)
    seq_length = 10
    train_sequences, train_labels = create_sequence_data(train_data, seq_length)
    val_sequences, val_labels = create_sequence_data(val_data, seq_length)

    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1

    learning_rates = [0.001, 0.01, 0.1]  # Различные значения learning rate
    batch_sizes = [32, 64, 128]  # Различные значения batch size

    num_epochs = 10

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Training with learning rate: {lr}, batch size: {batch_size}")
            model = LSTMModel(input_size, hidden_size, num_layers, output_size)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_dataset = TensorDataset(train_sequences, train_labels)
            val_dataset = TensorDataset(val_sequences, val_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs)
            # Здесь можно сохранять или анализировать результаты, например, выводить графики и т.д.
            print("Training completed.\n")