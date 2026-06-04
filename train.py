import torch
import torch.nn as nn
import argparse
import time
from models import MLP, CNN, PINN, physics_loss
from dataset import get_dataloaders
import numpy as np
import os

torch.manual_seed(42)
np.random.seed(42)

def train(model, train_loader, val_loader, optimizer, n_epochs, save_path, stats=None, lambda_physics=0.1, surrogate=None, lambda_surrogate=0.1):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    physics_losses = []

    for epoch in range(n_epochs):
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        train_physics = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)

            # Data loss
            data_loss = nn.MSELoss()(predictions, y_batch)

            # Add physics loss if PINN
            if stats is not None:
                phys_loss = physics_loss(predictions, stats)
                # Warm starting: gradually increase physics weight
                warmup_epochs = 50
                if epoch < warmup_epochs:
                    effective_lambda = 0.0
                else:
                    effective_lambda = lambda_physics * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
                loss = data_loss + effective_lambda * phys_loss
                train_physics += phys_loss.item()

                # Surrogate constraint
                if surrogate is not None:
                    surrogate.eval()
                    reconstructed = surrogate(predictions)
                    surrogate_loss = nn.MSELoss()(reconstructed, X_batch)
                    if epoch >= warmup_epochs:
                        loss = loss + lambda_surrogate * surrogate_loss
            else:
                loss = data_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_physics /= len(train_loader)
        train_losses.append(train_loss)

        if stats is not None:
            physics_losses.append(train_physics)

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                loss = nn.MSELoss()(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # --- Save best model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        # --- Print progress ---
        if (epoch+1) % 5 == 0:
            if stats is not None:
                effective_lambda_print = 0.0 if epoch < 50 else lambda_physics * (epoch - 50) / (n_epochs - 50)
                print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.6f} | Physics Loss: {train_physics:.6f} | Effective λ: {effective_lambda_print:.4f} | Val Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}")
                
    return train_losses, val_losses, physics_losses

def main():
    # Create directories
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/losses", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "cnn", "pinn"],
                        help="Which model to train: mlp, cnn, or pinn")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lambda_physics", type=float, default=1.0,
                        help="Physics loss weight (PINN only)")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, stats = get_dataloaders(batch_size=args.batch_size)

    # Build model
    print(f"Building {args.model.upper()}...")
    if args.model == "mlp":
        model = MLP()
        save_path = "data/models/mlp_best.pth"
        stats_for_physics = None
    elif args.model == "cnn":
        model = CNN()
        save_path = "data/models/cnn_best.pth"
        stats_for_physics = None
    elif args.model == "pinn":
        model = PINN()
        save_path = "data/models/pinn_best.pth"
        stats_for_physics = stats

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    print(f"Training {args.model.upper()} for {args.epochs} epochs...")
    start = time.time()
    train_losses, val_losses, physics_losses = train(
        model, train_loader, val_loader,
        optimizer, args.epochs, save_path,
        stats=stats_for_physics,
        lambda_physics=args.lambda_physics
    )

    # Save loss histories
    np.save(f"data/losses/{args.model}_train_losses.npy", np.array(train_losses))
    np.save(f"data/losses/{args.model}_val_losses.npy", np.array(val_losses))
    print(f"Loss histories saved to data/losses/{args.model}_train_losses.npy")

    # Save physics losses for PINN
    if args.model == "pinn" and len(physics_losses) > 0:
        np.save(f"data/losses/pinn_physics_losses.npy", np.array(physics_losses))
        print(f"Physics loss history saved to data/losses/pinn_physics_losses.npy")

    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best model saved to {save_path}")


if __name__ == "__main__":
    main()