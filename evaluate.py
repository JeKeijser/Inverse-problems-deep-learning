import torch
import numpy as np
import matplotlib.pyplot as plt
from models import MLP, CNN, PINN
from dataset import get_dataloaders
import os

def evaluate(model, test_loader, stats):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            all_predictions.append(predictions.numpy())
            all_targets.append(y_batch.numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)  # (1500, 4)
    all_targets = np.concatenate(all_targets, axis=0)          # (1500, 4)

    # Denormalize back to physical units
    y_min = stats["y_min"]
    y_max = stats["y_max"]
    all_predictions = all_predictions * (y_max - y_min) + y_min
    all_targets = all_targets * (y_max - y_min) + y_min

    return all_predictions, all_targets

def compute_metrics(predictions, targets):
    param_names = ["RI", "radius_x", "radius_y", "halo"]
    metrics = {}

    for i, name in enumerate(param_names):
        pred = predictions[:, i]
        targ = targets[:, i]

        # Mean Absolute Error
        mae = np.mean(np.abs(pred - targ))
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((pred - targ) ** 2))
        # R² score
        ss_res = np.sum((pred - targ) ** 2)
        ss_tot = np.sum((targ - np.mean(targ)) ** 2)
        r2 = 1 - ss_res / ss_tot

        metrics[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    return metrics

def print_metrics(metrics, model_name):
    print(f"\n--- {model_name} ---")
    print(f"{'Parameter':<12} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 45)
    for name, m in metrics.items():
        print(f"{name:<12} {m['MAE']:>10.2e} {m['RMSE']:>10.2e} {m['R2']:>10.4f}")

def plot_predictions(predictions, targets, model_name):
    param_names = ["RI", "radius_x (m)", "radius_y (m)", "halo (m)"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.2, s=5)
        # Perfect prediction line
        min_val = min(targets[:, i].min(), predictions[:, i].min())
        max_val = max(targets[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(name)
        ax.legend()

    plt.suptitle(f"{model_name} — Predicted vs True")
    plt.tight_layout()
    plt.savefig(f"data/plots/{model_name.lower()}_predictions.png")
    plt.show()


def plot_loss_curves(train_losses, val_losses, model_name, physics_losses=None):
    if physics_losses is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Total loss
        axes[0].plot(train_losses, label="Train Loss")
        axes[0].plot(val_losses, label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_title(f"{model_name} — Total Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Physics loss
        axes[1].plot(physics_losses, label="Physics Loss", color="red")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Physics Loss")
        axes[1].set_title(f"{model_name} — Physics Loss")
        axes[1].legend()
        axes[1].grid(True)

    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_losses, label="Train Loss")
        ax.plot(val_losses, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"{model_name} — Loss Curves")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"data/plots/{model_name.lower()}_loss_curves.png")
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def classification_accuracy(predictions, targets, threshold=1.405):
    # Convert RI predictions to filled/empty
    pred_filled = predictions[:, 0] > threshold
    true_filled = targets[:, 0] > threshold
    accuracy = np.mean(pred_filled == true_filled)
    return accuracy

def classification_f1(predictions, targets, threshold=1.405):
    # Classify as filled (1) or empty (0) based on RI threshold
    pred_filled = (predictions[:, 0] > threshold).astype(int)
    true_filled = (targets[:, 0] > threshold).astype(int)

    # True positives, false positives, false negatives
    TP = np.sum((pred_filled == 1) & (true_filled == 1))
    FP = np.sum((pred_filled == 1) & (true_filled == 0))
    FN = np.sum((pred_filled == 0) & (true_filled == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def plot_mean_loss_curves(train_arr, val_arr, model_name, physics_arr=None):
    epochs = np.arange(train_arr.shape[1])

    if physics_arr is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_loss = axes[0]
        ax_phys = axes[1]
    else:
        fig, ax_loss = plt.subplots(figsize=(8, 5))

    train_mean = train_arr.mean(axis=0)
    train_std  = train_arr.std(axis=0)
    val_mean   = val_arr.mean(axis=0)
    val_std    = val_arr.std(axis=0)

    ax_loss.plot(epochs, train_mean, label="Train Loss")
    ax_loss.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax_loss.plot(epochs, val_mean, label="Val Loss")
    ax_loss.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.set_title(f"{model_name} — Mean Loss Curves (±1 std)")
    ax_loss.legend()
    ax_loss.grid(True)

    if physics_arr is not None:
        phys_mean = physics_arr.mean(axis=0)
        phys_std  = physics_arr.std(axis=0)
        ax_phys.plot(epochs, phys_mean, label="Physics Loss", color="red")
        ax_phys.fill_between(epochs, phys_mean - phys_std, phys_mean + phys_std, alpha=0.2, color="red")
        ax_phys.set_xlabel("Epoch")
        ax_phys.set_ylabel("Physics Loss")
        ax_phys.set_title(f"{model_name} — Mean Physics Loss (±1 std)")
        ax_phys.legend()
        ax_phys.grid(True)

    plt.tight_layout()
    plt.savefig(f"data/plots/{model_name.lower()}_loss_curves.png")
    plt.show()

def main():
    os.makedirs("data/plots", exist_ok=True)
    print("Loading data...")
    _, _, test_loader, stats = get_dataloaders()

    seeds = [42, 123, 456]
    model_configs = {
        "MLP":  (MLP,  "data/models/mlp_best_seed{}.pth"),
        "CNN":  (CNN,  "data/models/cnn_best_seed{}.pth"),
        "PINN": (PINN, "data/models/pinn_best_seed{}.pth"),
    }

    for model_name, (model_class, path_template) in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}...")
        print(f"{'='*50}")

        all_predictions = []
        all_targets = []
        all_metrics = []
        all_accuracies = []
        all_precisions = []
        all_recalls = []
        all_f1s = []

        for seed in seeds:
            path = path_template.format(seed)
            if not os.path.exists(path):
                print(f"Model not found: {path}")
                continue

            model = model_class()
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            predictions, targets = evaluate(model, test_loader, stats)

            metrics = compute_metrics(predictions, targets)
            accuracy = classification_accuracy(predictions, targets)
            precision, recall, f1 = classification_f1(predictions, targets)

            all_predictions.append(predictions)
            all_targets.append(targets)
            all_metrics.append(metrics)
            all_accuracies.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)

            print(f"\nSeed {seed}:")
            print_metrics(metrics, f"{model_name} seed {seed}")
            print(f"  Accuracy:  {accuracy*100:.1f}%")
            print(f"  Precision: {precision*100:.1f}%")
            print(f"  Recall:    {recall*100:.1f}%")
            print(f"  F1-score:  {f1*100:.1f}%")

        # Compute mean and std across seeds
        print(f"\n--- {model_name} Mean ± Std ---")
        param_names = ["RI", "radius_x", "radius_y", "halo"]
        print(f"{'Parameter':<12} {'R² Mean':>10} {'R² Std':>10} {'MAE Mean':>12} {'MAE Std':>10}")
        print("-" * 55)
        for param in param_names:
            r2_values  = [m[param]['R2']  for m in all_metrics]
            mae_values = [m[param]['MAE'] for m in all_metrics]
            print(f"{param:<12} {np.mean(r2_values):>10.4f} {np.std(r2_values):>10.4f} {np.mean(mae_values):>12.2e} {np.std(mae_values):>10.2e}")

        print(f"\n{model_name} Classification (threshold RI=1.405):")
        print(f"  Accuracy:  {np.mean(all_accuracies)*100:.1f}% ± {np.std(all_accuracies)*100:.1f}%")
        print(f"  Precision: {np.mean(all_precisions)*100:.1f}% ± {np.std(all_precisions)*100:.1f}%")
        print(f"  Recall:    {np.mean(all_recalls)*100:.1f}% ± {np.std(all_recalls)*100:.1f}%")
        print(f"  F1-score:  {np.mean(all_f1s)*100:.1f}% ± {np.std(all_f1s)*100:.1f}%")

        # Plot using seed 42 as representative (if available)
        if all_predictions:
            plot_predictions(all_predictions[0], all_targets[0], model_name)
        else:
            print(f"No models found for {model_name}, skipping plots")

        # Plot mean loss curves across all seeds
        all_train_losses = []
        all_val_losses_plot = []
        all_physics_losses_plot = []

        for seed in seeds:
            t_path = f"data/losses/{model_name.lower()}_best_seed{seed}_train_losses.npy"
            v_path = f"data/losses/{model_name.lower()}_best_seed{seed}_val_losses.npy"
            p_path = f"data/losses/{model_name.lower()}_best_seed{seed}_physics_losses.npy"

            if os.path.exists(t_path):
                all_train_losses.append(np.load(t_path))
                all_val_losses_plot.append(np.load(v_path))
                if os.path.exists(p_path):
                    all_physics_losses_plot.append(np.load(p_path))

        if all_train_losses:
            max_len = max(len(l) for l in all_train_losses)
            def pad(arr): return np.pad(arr, (0, max_len - len(arr)), constant_values=arr[-1])

            train_arr = np.array([pad(l) for l in all_train_losses])
            val_arr   = np.array([pad(l) for l in all_val_losses_plot])
            phys_arr  = np.array([pad(l) for l in all_physics_losses_plot]) if all_physics_losses_plot else None

            plot_mean_loss_curves(train_arr, val_arr, model_name, physics_arr=phys_arr)


if __name__ == "__main__":
    main()