import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split


def load_data(file_path: str) -> (torch.tensor, torch.tensor):
    df = pd.read_csv(file_path,delimiter=",")
    df.columns = ['Дата', 'Курс']
    df['Курс'] = pd.to_numeric(df['Курс'], errors="coerce")

    invalid_values = df[df['Курс'].isnull()]
    if not invalid_values.empty:
        df = df.dropna(subset=['Курс'])
        
    timestamps = pd.to_datetime(df['Дата']).astype('int64') // 10**9
    #df['Курс'] = (df['Курс'] - df['Курс'].min()) / (df['Курс'].max() - df['Курс'].min())

    dates_tensor = torch.tensor(timestamps.values, dtype=torch.float).view(-1, 1)
    exchange_rates_tensor = torch.tensor(df['Курс'].values, dtype=torch.float).view(-1, 1)
    
    return (dates_tensor,exchange_rates_tensor)


def separation_sample(dates_tensor: torch.tensor,
                      exchange_rates_tensor: torch.tensor
                      ) -> (DataLoader,DataLoader,DataLoader):
    dataset = TensorDataset(dates_tensor,exchange_rates_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = (len(dataset) - train_size) // 2
    val_size = len(dataset) - train_size - test_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    return (train_loader,val_loader,test_loader)