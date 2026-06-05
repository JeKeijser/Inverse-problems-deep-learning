import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.network(x)
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                            
            nn.Dropout(p=0.5),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                            
            nn.Dropout(p=0.5),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                            
            nn.Dropout(p=0.5),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)
    
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          
            nn.Dropout(p=0.5),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                            
            nn.Dropout(p=0.5),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                           
            nn.Dropout(p=0.5),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)

def physics_loss(predictions, stats):
   
    # Denormalize predictions back to physical units
    y_min = torch.tensor(stats["y_min"], dtype=torch.float32).to(predictions.device)
    y_max = torch.tensor(stats["y_max"], dtype=torch.float32).to(predictions.device)
    phys = predictions * (y_max - y_min) + y_min

    RI       = phys[:, 0]
    radius_x = phys[:, 1]
    radius_y = phys[:, 2]
    halo     = phys[:, 3]

    # Constraint 1
    ri_low  = torch.relu(1.34 - RI)   # penalty if RI < 1.34
    ri_high = torch.relu(RI - 1.56)   # penalty if RI > 1.56
    ri_penalty = (ri_low + ri_high).mean()

    # Constraint 2
    dist_empty  = (RI - 1.37) ** 2
    dist_filled = (RI - 1.45) ** 2
    bimodal_penalty = torch.min(dist_empty, dist_filled).mean()

    # Constraint 3
    r_low_x  = torch.relu(100e-9 - radius_x)
    r_high_x = torch.relu(radius_x - 400e-9)
    r_low_y  = torch.relu(100e-9 - radius_y)
    r_high_y = torch.relu(radius_y - 400e-9)
    radius_penalty = (r_low_x + r_high_x + r_low_y + r_high_y).mean()

    # Constraint 4
    halo_penalty = torch.relu(-halo).mean()

    # Total physics loss
    total = ri_penalty + 1.0 * bimodal_penalty + radius_penalty + halo_penalty

    return total
    
if __name__ == "__main__":
    dummy_input = torch.randn(32, 1, 64, 64)

    # Test MLP
    mlp = MLP()
    print("MLP architecture:")
    print(mlp)
    mlp_out = mlp(dummy_input)
    print("MLP input shape:", dummy_input.shape)
    print("MLP output shape:", mlp_out.shape)
    print()

    # Test CNN
    cnn = CNN()
    print("CNN architecture:")
    print(cnn)
    cnn_out = cnn(dummy_input)
    print("CNN input shape:", dummy_input.shape)
    print("CNN output shape:", cnn_out.shape)
    print()

    # Test PINN
    pinn = PINN()
    print("PINN architecture:")
    print(pinn)
    pinn_out = pinn(dummy_input)
    print("PINN input shape:", dummy_input.shape)
    print("PINN output shape:", pinn_out.shape)
    print()

    # Test physics loss
    dummy_stats = {
        "y_min": [1.34, 100e-9, 100e-9, 0.0],
        "y_max": [1.56, 400e-9, 400e-9, 500e-9]
    }
    dummy_predictions = torch.rand(32, 4)
    ploss = physics_loss(dummy_predictions, dummy_stats)
    print("Physics loss test:", ploss.item())
