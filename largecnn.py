import torch
import torch.nn as nn

class LargeCNN(nn.Module):
    def __init__(self):
        super(LargeCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),   # ↑ 16 → 32
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1, 1),  # ↑ 32 → 64
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),  # ↑ neurons
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)