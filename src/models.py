import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(45, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.fc(x)

class MITInspired(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(45, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.view(-1, 15, 3)  # Updated to 15 atoms
        r = torch.norm(x, dim=2).mean(dim=1, keepdim=True)
        return self.fc(r.repeat(1, 45))  # Updated to 45 features

class ASAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.meta_syms = nn.Parameter(torch.tensor([-0.6, -0.3, 0.0, 0.3, 0.6]) * 0.4)
        self.fc = nn.Sequential(
            nn.Linear(225, 128),  # 5 syms * 15 atoms * 3 coords = 225
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(-1, 15, 3)  # Reshape to 15 atoms x 3
        r = torch.norm(x, dim=2).unsqueeze(2)
        theta = torch.acos(x[:, :, 2] / (r.squeeze(2) + 1e-6)).unsqueeze(2)
        phi = torch.atan2(x[:, :, 1], x[:, :, 0]).unsqueeze(2)
        adaptive_features = []
        for sym in self.meta_syms:
            theta_adapt = sym * theta
            phi_adapt = sym * phi
            adaptive = torch.cat([r * torch.sin(theta_adapt) * torch.cos(phi_adapt),
                                  r * torch.sin(theta_adapt) * torch.sin(phi_adapt),
                                  r * torch.cos(theta_adapt)], dim=2).view(x.size(0), -1)
            adaptive_features.append(adaptive)
        adaptive = torch.cat(adaptive_features, dim=1)  # 5 syms * 45 features
        return self.fc(adaptive)