import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("data/plots", exist_ok=True)

for model_name in ["mlp", "cnn", "pinn"]:
    train_losses = np.load(f"data/losses/{model_name}_train_losses.npy")
    val_losses = np.load(f"data/losses/{model_name}_val_losses.npy")
    
    if model_name == "pinn":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(train_losses, label="Train Loss")
        axes[0].plot(val_losses, label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_title("PINN — Loss Curves")
        axes[0].legend()
        axes[0].grid(True)
        phys = np.load("data/losses/pinn_physics_losses.npy")
        axes[1].plot(phys, color="red", label="Physics Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Physics Loss")
        axes[1].set_title("PINN — Physics Loss")
        axes[1].legend()
        axes[1].grid(True)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_losses, label="Train Loss")
        ax.plot(val_losses, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"{model_name.upper()} — Loss Curves")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"data/plots/{model_name}_loss_curves.png")
    plt.show()
    print(f"Saved {model_name} loss curves!")