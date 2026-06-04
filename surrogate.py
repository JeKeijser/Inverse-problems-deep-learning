import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.utils.data import DataLoader
from models import SurrogateModel
from dataset import LNPDataset, load_and_normalize

torch.manual_seed(42)
np.random.seed(42)

def train_surrogate(n_epochs=100, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data — note: X is target, y is input (reversed!)
    X_norm, y_norm, stats = load_and_normalize()

    # Create dataset with y as input and X as target
    train_size = int(0.85 * len(X_norm))
    
    X_train = X_norm[:train_size]
    y_train = y_norm[:train_size]
    X_val   = X_norm[train_size:]
    y_val   = y_norm[train_size:]

    # Custom dataset for surrogate — input is params, target is image
    class SurrogateDataset(torch.utils.data.Dataset):
        def __init__(self, params, images):
            self.params = torch.tensor(params, dtype=torch.float32)
            self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        
        def __len__(self):
            return len(self.params)
        
        def __getitem__(self, idx):
            return self.params[idx], self.images[idx]

    train_dataset = SurrogateDataset(y_train, X_train)
    val_dataset   = SurrogateDataset(y_val, X_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # Build model
    model = SurrogateModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"Training surrogate for {n_epochs} epochs...")
    start = time.time()

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for params, images in train_loader:
            params = params.to(device)
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed = model(params)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for params, images in val_loader:
                params = params.to(device)
                images = images.to(device)
                reconstructed = model(params)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("data/models", exist_ok=True)
            torch.save(model.state_dict(), "data/models/surrogate_best.pth")

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}")

    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.1f}s")
    
    # Save loss histories
    os.makedirs("data/losses", exist_ok=True)
    np.save("data/losses/surrogate_train_losses.npy", np.array(train_losses))
    np.save("data/losses/surrogate_val_losses.npy", np.array(val_losses))
    print(f"Best val loss: {best_val_loss:.6f}")

    return model, train_losses, val_losses


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    model, train_losses, val_losses = train_surrogate(n_epochs=5)
    
    # Load some real data
    X_norm, y_norm, stats = load_and_normalize()
    
    # Compare real vs reconstructed
    device = torch.device("cpu")
    model.eval()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    with torch.no_grad():
        for i in range(4):
            params = torch.tensor(y_norm[i], dtype=torch.float32).unsqueeze(0)
            reconstructed = model(params).squeeze().numpy()
            real = X_norm[i]
            
            axes[0, i].imshow(real, cmap='gray')
            axes[0, i].set_title(f"Real {i+1}")
            axes[1, i].imshow(reconstructed, cmap='gray')
            axes[1, i].set_title(f"Reconstructed {i+1}")
    
    plt.tight_layout()
    plt.savefig("data/plots/surrogate_test.png")
    plt.show()