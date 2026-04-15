import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),   # ↓ 16 → 8
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, 1, 1),  # ↓ 32 → 16
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 16 * 16, 64),  # ↓ neurons
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)