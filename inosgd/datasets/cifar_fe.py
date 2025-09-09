import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from inosgd.utils.saveload import load


class _IndexedCifarFeatures(Dataset):
    def __init__(self, train=True, indexed=True):
        if not train:
            self.data = torch.Tensor(np.load("transfer/features/test_data.npy")).float()
            self.targets = torch.Tensor(np.load("transfer/features/test_targets.npy")).reshape(-1).int()
        else:
            indices = torch.arange(50000)
            self.data = torch.Tensor(np.load("transfer/features/train_data.npy")).float()
            if not indexed:
                self.targets = torch.Tensor(np.load("transfer/features/train_targets.npy")).reshape(-1).int()
            else:
                self.targets = torch.stack([torch.Tensor(np.load("transfer/features/train_targets.npy")).reshape(-1).int(), indices], dim=1)
        

    def __len__(self):
        return len(self.targets)
    

    def __getitem__(self, index):
        datum, target = self.data[index], self.targets[index].int()
        return datum, target


def load_cifar_features(which: str):
    if which == 'train':
        return _IndexedCifarFeatures(train=True, indexed=True)
    elif which == 'test':
        return _IndexedCifarFeatures(train=False, indexed=False)
    else:
        raise ValueError("Can only choose between 'train' and 'test'.")
    