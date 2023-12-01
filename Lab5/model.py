from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_data(file_path: str) -> torch.Tensor:
    """
    Loads data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - torch.Tensor: Data tensor.
    """
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data['Value'] = (data['Value'] - data['Value'].min()) / (data['Value'].max() - data['Value'].min())
    return torch.tensor(data['Value'].values).float()


def split_data(all_data: torch.Tensor, train_ratio: float = 0.8, val_ratio: float = 0.1) -> (torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset):
    """
    Splits data into training, validation, and test sets.

    Parameters:
    - all_data (torch.Tensor): All data.
    - train_ratio (float): Ratio of training data.
    - val_ratio (float): Ratio of validation data.

    Returns:
    - torch.utils.data.Dataset: Training dataset.
    - torch.utils.data.Dataset: Validation dataset.
    - torch.utils.data.Dataset: Test dataset.
    """
    train_size = int(train_ratio * len(all_data))
    val_size = int(val_ratio * len(all_data))
    test_size = len(all_data) - train_size - val_size

    train_data, val_data, test_data = random_split(all_data, [train_size, val_size, test_size])
    return train_data, val_data, test_data


def create_sequences(data: torch.Tensor, seq_length: int) -> (torch.Tensor, torch.Tensor):
    """
    Creates sequences from data.

    Parameters:
    - data (torch.Tensor): Input data.
    - seq_length (int): Length of the sequence.

    Returns:
    - torch.Tensor: Sequences.
    - torch.Tensor: Labels.
    """
    sequences = []
    labels = []

    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append(seq)
        labels.append(label)

    return torch.stack(sequences), torch.stack(labels)


def train_lstm_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epochs: int) -> (list,list):
    """
    Trains the LSTM model.

    Parameters:
    - model (nn.Module): LSTM model.
    - train_loader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validate data.
    - criterion (nn.Module): Loss function.
    - optimizer (optim.Optimizer): Optimizer.
    - epochs (int): Number of epochs.
    
    Returns:
    - (list,list): Lists of loss value in train and validation mode.
    """
    epoch_train_losses = []
    epoch_val_losses = []
    for epoch in range(epochs):
        list_of_losses = []
        for inputs, labels in train_loader:
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(-1).float())
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            list_of_losses.append(loss.item())
        epoch_train_losses.append(sum(list_of_losses)/len(list_of_losses))
        epoch_val_losses.append(evaluate_model(model, val_loader))
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {epoch_train_losses[epoch]}\t Validation loss: {epoch_val_losses[epoch]}')  
        
    return epoch_train_losses,epoch_val_losses


def plot_loss_history(train_losses: list, val_losses: list, epochs_list: list, lr: float, batch_size: int, ax=None) -> None:
    """
    Plots the loss history.

    Parameters:
    - train_losses (list): List of training loss values.
    - val_losses (list): List of validation loss values.
    - epochs_list (list): List of epochs.
    - lr (float): Learning rate.
    - batch_size (int): Batch size.
    - ax (Axes, optional): Axes object to draw the plot. If not provided, a new plot will be created.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(epochs_list, train_losses, label=f'Train Loss (LR: {lr}, Batch Size: {batch_size})')
    ax.plot(epochs_list, val_losses, label=f'Validation Loss (LR: {lr}, Batch Size: {batch_size})')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training and Validation Loss (LR: {lr}, Batch Size: {batch_size})')
    ax.legend()

def evaluate_model(model: nn.Module, data_loader: DataLoader) -> float:
    """
    Evaluates the model on data.

    Parameters:
    - model (nn.Module): Model.
    - data_loader (DataLoader): DataLoader for data.

    Returns:
    - float: loss value.
    """
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        model.eval()
        for inputs, labels in data_loader:
            outputs = model(inputs.unsqueeze(-1).float())
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()
    return total_loss


def test_model(model: nn.Module, test_loader: DataLoader) -> float:
    """
    Tests the model on test data.

    Parameters:
    - model (nn.Module): Model.
    - test_loader (DataLoader): DataLoader for test data.

    Returns:
    - float: Loss value.
    """
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.unsqueeze(-1).float())
            loss = criterion(outputs.squeeze(), labels.float())
    return loss.item()


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def main(file_path: str = 'dataset/dataset.csv', seq_length: int = 10, input_size: int = 1, hidden_size: int = 64, output_size: int = 1,
         learning_rates: list = [0.001, 0.01, 0.1], batch_sizes: list = [32, 64, 128], epochs: int = 10):
    """
    Main function to execute the entire process.

    Parameters:
    - file_path (str): Path to the CSV file.
    - seq_length (int): Length of the sequence.
    - input_size (int): Model input size.
    - hidden_size (int): Model hidden layer size.
    - output_size (int): Model output size.
    - learning_rates (list): List of learning rate values.
    - batch_sizes (list): List of batch sizes.
    - epochs (int): Number of training epochs.
    """
    all_data = load_data(file_path)
    train_data, val_data, test_data = split_data(all_data)

    X_train, y_train = create_sequences(train_data, seq_length)
    X_val, y_val = create_sequences(val_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    epochs_list = []
    
    fig, axes = plt.subplots(len(learning_rates), len(batch_sizes), figsize=(12, 12))
    axes = axes.flatten()
    
    for i, lr in enumerate(learning_rates):
        for j, batch_size in enumerate(batch_sizes):
            model = SimpleLSTM(input_size, hidden_size, output_size)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
            
            print(f"Learning rate: {lr}, batch size: {batch_size}")
            train_losses,val_losses = train_lstm_model(model, train_loader, val_loader, criterion, optimizer, epochs)

            epochs_list = list(range(1, epochs + 1))
            
            plot_loss_history(train_losses, val_losses, epochs_list, lr, batch_size, ax=axes[i * len(batch_sizes) + j])

    plt.tight_layout()
    plt.show()
    
    test_loss = test_model(model, DataLoader(TensorDataset(X_test, y_test), batch_size=1))
    print(f'Test Loss: {test_loss}')

    torch.save(model.state_dict(), 'trained_model.pth')

    model = SimpleLSTM(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()

    with torch.no_grad():
        test_inputs, test_labels = X_test, y_test
        test_outputs = model(test_inputs.unsqueeze(-1).float())

        month_index = 1

        predicted_value = test_outputs[month_index].item()
        true_value = test_labels[month_index].item()

        print(f'Predicted Value: {predicted_value}')
        print(f'True Value: {true_value}')

if __name__ == "__main__":
    main()