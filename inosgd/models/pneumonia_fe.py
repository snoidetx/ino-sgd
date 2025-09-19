import torch.nn as nn


class PneumoniaFENet(nn.Module):
    def __init__(self, input_dim=1000, hidden_dim=512, num_classes=3):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, num_classes))


    def forward(self, x):
        return self.fc(x)
