import torch.nn as nn


class CifarFENet(nn.Module):
    def __init__(self, input_dim=4096, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)


    def forward(self, x):
        return self.fc(x)
