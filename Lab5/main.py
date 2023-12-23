# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


def generate_time_lags(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """
    Generate time lag features for a given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the time series data.
        n_lags (int): The number of lag features to generate.

    Returns:
        pd.DataFrame: The DataFrame with lag features.
    """
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n


def feature_label_split(df: pd.DataFrame, target_col: str) -> tuple:
    """
    Split the DataFrame into features and labels.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        target_col (str): The column name of the target variable.

    Returns:
        tuple: A tuple containing features (X) and labels (y).
    """
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y


def train_val_test_split(df: pd.DataFrame, target_col: str, test_ratio: float) -> tuple:
    """
    Split the DataFrame into training, validation, and test sets.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        target_col (str): The column name of the target variable.
        test_ratio (float): The ratio of the test set.

    Returns:
        tuple: A tuple containing training, validation, and test sets for features and labels.
    """
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_scaler(scaler: str) -> StandardScaler:
    """
    Get a scaler object based on the specified scaler type.

    Args:
        scaler (str): The type of scaler.

    Returns:
        sklearn.preprocessing: The scaler object.
    """
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()


class RNNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layer_dim: int, output_dim: int, dropout_prob: float):
        """
        Initialize an RNN model.

        Args:
            input_dim (int): The number of nodes in the input layer.
            hidden_dim (int): The number of nodes in each layer.
            layer_dim (int): The number of layers in the network.
            output_dim (int): The number of nodes in the output layer.
            dropout_prob (float): The probability of nodes being dropped out.
        """
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RNN model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, h0 = self.rnn(x, h0.detach())
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class Optimization:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer):
        """
        Initialize the optimization object.

        Args:
            model (nn.Module): The neural network model.
            loss_fn (nn.Module): The loss function.
            optimizer (optim.Optimizer): The optimizer.
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Perform one step of training.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The target tensor.

        Returns:
            float: The training loss.
        """
        self.model.train()
        self.optimizer.zero_grad()
        yhat = self.model(x)
        loss = self.loss_fn(y, yhat)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, train_loader: DataLoader, val_loader: DataLoader, batch_size: int = 64,
              n_epochs: int = 50, n_features: int = 1):
        """
        Train the model.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            batch_size (int): Batch size.
            n_epochs (int): Number of epochs.
            n_features (int): Number of input features.
        """
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) or (epoch % 50 == 0):
                print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")

    def evaluate(self, test_loader: DataLoader, batch_size: int = 1, n_features: int = 1) -> tuple:
        """
        Evaluate the model on the test set.

        Args:
            test_loader (DataLoader): DataLoader for the test set.
            batch_size (int): Batch size.
            n_features (int): Number of input features.

        Returns:
            tuple: A tuple containing predictions and true values.
        """
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())
        return predictions, values

    def plot_losses(self, lr: float, bs: int):
        """
        Plot training and validation losses and save the plot with LR and BS in the title.

        Args:
            lr (float): Learning rate.
            bs (int): Batch size.
        """
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Losses (LR={lr}, BS={bs})")
        plt.savefig(f"losses_LR_{lr}_BS_{bs}.png")  # Save the plot with LR and BS in the filename
        plt.show()
        plt.close()


def inverse_transform(scaler, df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Inverse transform scaled columns using the specified scaler.

    Args:
        scaler: The scaler object used for transformation.
        df (pd.DataFrame): The DataFrame containing the scaled columns.
        columns (list): List of column names to inverse transform.

    Returns:
        pd.DataFrame: The DataFrame with inverse transformed columns.
    """
    df[columns] = scaler.inverse_transform(df[columns])
    return df


def format_predictions(predictions, values, df_test: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Format predictions and true values into a DataFrame.

    Args:
        predictions: Predicted values.
        values: True values.
        df_test (pd.DataFrame): The original DataFrame for index information.
        scaler: Scaler object used for transformation.

    Returns:
        pd.DataFrame: Formatted DataFrame with 'value' and 'prediction' columns.
    """
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, ["value", "prediction"])
    return df_result


def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate evaluation metrics for a DataFrame with 'value' and 'prediction' columns.

    Args:
        df (pd.DataFrame): DataFrame with 'value' and 'prediction' columns.

    Returns:
        dict: Dictionary containing calculated metrics ('mae', 'rmse', 'r2').
    """
    result_metrics = {'mae': mean_absolute_error(df.value, df.prediction),
                      'rmse': mean_squared_error(df.value, df.prediction) ** 0.5,
                      'r2': r2_score(df.value, df.prediction)}

    print("Mean Absolute Error:       ", result_metrics["mae"])
    print("Root Mean Squared Error:   ", result_metrics["rmse"])
    print("R^2 Score:                 ", result_metrics["r2"])
    return result_metrics


def build_baseline_model(df: pd.DataFrame, test_ratio: float, target_col: str) -> pd.DataFrame:
    """
    Build a baseline linear regression model and make predictions.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        test_ratio (float): Ratio of the test set.
        target_col (str): Column name of the target variable.

    Returns:
        pd.DataFrame: DataFrame with true and predicted values.
    """
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    result = pd.DataFrame(y_test)
    result["prediction"] = prediction
    result = result.sort_index()

    return result


def check_balance(y: pd.Series) -> bool:
    """
    Check the balance of a binary target variable.

    Args:
        y (pd.Series): The target variable.

    Returns:
        bool: True if the target variable is balanced, False otherwise.
    """
    class_counts = y.value_counts()
    balance_check = abs(class_counts[0] - class_counts[1]) / len(y) < 0.1
    return balance_check


if __name__ == '__main__':
    device = "cpu"

    df = pd.read_csv('weather.csv')

    df = df.set_index(['Дата'])
    df = df.rename(columns={'Температура': 'value'})

    df.index = pd.to_datetime(df.index)

    input_dim = 100

    df_timelags = generate_time_lags(df, input_dim)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_timelags, 'value', 0.2)

    # Проверяем балансировку для y_train, y_val, y_test
    print("Balance Check - Train Set:", check_balance(y_train))
    print("Balance Check - Validation Set:", check_balance(y_val))
    print("Balance Check - Test Set:", check_balance(y_test))

    scaler = get_scaler('minmax')

    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_val_arr = scaler.transform(y_val)
    y_test_arr = scaler.transform(y_test)

    batch_size = 64

    train_features = torch.Tensor(X_train_arr)
    train_targets = torch.Tensor(y_train_arr)
    val_features = torch.Tensor(X_val_arr)
    val_targets = torch.Tensor(y_val_arr)
    test_features = torch.Tensor(X_test_arr)
    test_targets = torch.Tensor(y_test_arr)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    input_dim = len(X_train.columns)
    output_dim = 1
    hidden_dim = 64
    layer_dim = 3
    batch_size = 64
    dropout = 0.2
    n_epochs = 20
    learning_rate = 1e-3
    weight_decay = 1e-6

    model_params = {'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'layer_dim': layer_dim,
                    'output_dim': output_dim,
                    'dropout_prob': dropout}

    model = RNNModel(**model_params)

    loss_fn = nn.MSELoss(reduction="mean")

    learning_rates = [1e-3, 1e-4, 1e-5]
    batch_sizes = [16, 32, 64]

    results = {}
    # experiments with learning_rates and batch_sizes
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\nExperiment with Learning Rate={lr} and Batch Size={bs}")

            model = RNNModel(**model_params)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)

            train_loader = DataLoader(train, batch_size=bs, shuffle=False, drop_last=True)
            val_loader = DataLoader(val, batch_size=bs, shuffle=False, drop_last=True)
            test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

            opt.train(train_loader, val_loader, batch_size=bs, n_epochs=n_epochs, n_features=input_dim)

            predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)
            df_result = format_predictions(predictions, values, X_test, scaler)

            experiment_name = f"LR={lr}_BS={bs}"
            results[experiment_name] = {
                'metrics': calculate_metrics(df_result),
                'model': model.state_dict(),
            }

            opt.plot_losses(lr=lr, bs=bs)
            torch.save(opt.model.state_dict(), f"trained_model_LR_{lr}_BS_{bs}.pth")
            plt.show()
    for experiment_name, metrics_dict in results.items():
        print(f"\nMetrics for experiment {experiment_name}:")
        for metric_name, metric_value in metrics_dict['metrics'].items():
            print(f"{metric_name}: {metric_value}")

    # loading ready model

    model_after_reload = RNNModel(**model_params)

    loaded_model_path = "trained_model_LR_0.0001_BS_16.pth"  # Укажите путь к сохраненным весам
    model_after_reload.load_state_dict(torch.load(loaded_model_path))
    model_after_reload.eval()

    with torch.no_grad():
        test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
        predictions_after_reload, values_after_reload = opt.evaluate(test_loader_one, batch_size=1,
                                                                     n_features=input_dim)
        df_result_after_reload = format_predictions(predictions_after_reload, values_after_reload, X_test, scaler)

    plt.plot(df_result_after_reload['value'], label='True Values')
    plt.plot(df_result_after_reload['prediction'], label='Predictions')
    plt.legend()
    plt.title('True Values vs Predictions after Reload')
    plt.show()
