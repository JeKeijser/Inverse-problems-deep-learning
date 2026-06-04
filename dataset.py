import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_and_normalize(data_path="data/raw/"):
    # Load raw data
    X = np.load(data_path + "X.npy")      # (10000, 64, 64)
    y = np.load(data_path + "y.npy")      # (10000, 4)

    # Normalize images to [0, 1]
    X_min = X.min()
    X_max = X.max()
    X_norm = (X - X_min) / (X_max - X_min)

    # Normalize labels to [0, 1] per parameter
    y_min = y.min(axis=0)
    y_max = y.max(axis=0)
    y_norm = (y - y_min) / (y_max - y_min)

    stats = {
        "X_min": X_min, "X_max": X_max,
        "y_min": y_min, "y_max": y_max
    }

    return X_norm, y_norm, stats

class LNPDataset(Dataset):
    def __init__(self, X, y):
        # Add channel dimension and convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 64, 64)
        self.y = torch.tensor(y, dtype=torch.float32)                # (N, 4)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(data_path="data/raw/", batch_size=32, val_split=0.15, test_split=0.15):
    X_norm, y_norm, stats = load_and_normalize(data_path)

    # Dataset sizes
    n = len(X_norm)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    # Split
    X_train, y_train = X_norm[:n_train], y_norm[:n_train]
    X_val,   y_val   = X_norm[n_train:n_train+n_val], y_norm[n_train:n_train+n_val]
    X_test,  y_test  = X_norm[n_train+n_val:], y_norm[n_train+n_val:]

    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Create datasets
    train_dataset = LNPDataset(X_train, y_train)
    val_dataset   = LNPDataset(X_val, y_val)
    test_dataset  = LNPDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, stats

if __name__ == "__main__":
    train_loader, val_loader, test_loader, stats = get_dataloaders()

    # Check a batch
    X_batch, y_batch = next(iter(train_loader))
    print("X batch shape:", X_batch.shape)  # (32, 1, 64, 64)
    print("y batch shape:", y_batch.shape)  # (32, 4)
    print("X range:", X_batch.min().item(), X_batch.max().item())
    print("y range:", y_batch.min().item(), y_batch.max().item())