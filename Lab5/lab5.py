
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None, label_mapping=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label_str = self.labels[idx]
        label = self.label_mapping[label_str] if self.label_mapping else int(
            label_str)

        return img, torch.tensor(label)


def load_dataset(csv_path: str, train_size=0.8, val_size=0.1, test_size=0.1) -> tuple:
    """Function for uploading data from an annotation file to a list"""
    try:
        dframe = pd.read_csv(
            csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"]
        )
        img_list = dframe["Absolute path"].tolist()
        labels = dframe["Class"].tolist()

        print(f"Original dataset size: {len(img_list)}")

        if not img_list or not labels:
            raise ValueError("Empty dataset: No images or labels found.")

        img_list, labels = list(img_list), list(labels)

        combined = list(zip(img_list, labels))
        random.seed(42)
        random.shuffle(combined)
        img_list[:], labels[:] = zip(*combined)

        img_train, labels_train, img_val, labels_val, img_test, labels_test = split_dataset(
            img_list, labels, train_size=train_size, val_size=val_size, test_size=test_size
        )

        print(f"Training dataset size: {len(img_train)}")
        print(f"Validation dataset size: {len(img_val)}")
        print(f"Test dataset size: {len(img_test)}")

        return img_train, labels_train, img_val, labels_val, img_test, labels_test
    except FileNotFoundError:
        print(f"Error: File not found at path '{csv_path}'")
        return [], [], [], [], [], []
    except pd.errors.EmptyDataError:
        print(f"Error: Empty file at path '{csv_path}'")
        return [], [], [], [], [], []
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return [], [], [], [], [], []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], [], [], [], [], []


def split_dataset(img_list, labels, train_size=0.8, val_size=0.1, test_size=0.1):
    """Split the dataset into training, validation, and test sets"""
    total_size = len(img_list)

    print(f"Total dataset size: {total_size}")

    train_size = int(total_size * train_size)
    val_size = int(total_size * val_size)
    test_size = int(total_size * test_size)

    print(f"Training dataset size: {train_size}")
    print(f"Validation dataset size: {val_size}")
    print(f"Test dataset size: {test_size}")

    if train_size <= 0:
        raise ValueError("Not enough samples for training.")

    combined = list(zip(img_list, labels))
    random.seed(42)
    random.shuffle(combined)
    img_list[:], labels[:] = zip(*combined)

    img_val, labels_val = img_list[:val_size], labels[:val_size]
    img_test, labels_test = img_list[val_size:val_size +
                                     test_size], labels[val_size:val_size + test_size]
    img_train, labels_train = img_list[val_size + test_size:val_size + test_size +
                                       train_size], labels[val_size + test_size:val_size + test_size + train_size]

    return img_train, labels_train, img_val, labels_val, img_test, labels_test


def calculate_accuracy(predictions, true_labels):
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    total = len(predictions)
    return correct / total


def plot_training_results(train_losses, val_losses, val_accuracies, learning_rate, batch_size):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy',
             marker='o', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.suptitle(f'Learning Rate: {learning_rate}, Batch Size: {batch_size}')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        for images, labels in train_loader:
            images = torch.stack([img.to(device) for img in images])
            labels = torch.as_tensor(
                labels, dtype=torch.long).clone().detach().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            predictions = []
            true_labels = []
            for images, labels in val_loader:
                images = torch.stack([img.to(device) for img in images])
                labels = torch.as_tensor(
                    labels, dtype=torch.long).clone().detach().to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            accuracy = calculate_accuracy(predictions, true_labels)

            print(
                f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

            val_losses.append(val_loss)
            val_accuracies.append(accuracy)

    return train_losses, val_losses, val_accuracies


def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_predictions = []
        test_true_labels = []
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())

    test_accuracy = calculate_accuracy(test_predictions, test_true_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")


def main(csv_path, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_train, labels_train, img_val, labels_val, img_test, labels_test = load_dataset(
        csv_path)

    unique_labels = set(labels_train + labels_val + labels_test)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            print(
                f"\nExperiment: Learning Rate = {learning_rate}, Batch Size = {batch_size}")

            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])

            train_dataset = CustomDataset(
                img_train, labels_train, transform, label_mapping)
            val_dataset = CustomDataset(
                img_val, labels_val, transform, label_mapping)
            test_dataset = CustomDataset(
                img_test, labels_test, transform, label_mapping)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

            model = SimpleCNN(num_classes=len(unique_labels)).to(device)

            train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, device,
                                                                   num_epochs=num_epochs, learning_rate=learning_rate)

            plot_training_results(train_losses, val_losses,
                                  val_accuracies, learning_rate, batch_size)

            evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main("annotation.csv")
