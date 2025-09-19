import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from inosgd.utils.saveload import load


class _IndexedPneumoniaFeatures(Dataset):
    PATH = "dataset/pneumonia/"

    def __init__(self, train=True, indexed=True):
        if not train:
            self.data = load(cls.PATH + "test_data.pkl").float()
            self.targets = load(cls.PATH + "test_targets.pkl").reshape(-1).int()
        else:
            indices = torch.arange(50000)
            self.data = load(cls.PATH + "train_data.pkl").float()
            if not indexed:
                self.targets = load(cls.PATH + "train_targets.pkl").reshape(-1).int()
            else:
                self.targets = torch.stack([load(cls.PATH + "train_targets.pkl").reshape(-1).int(), indices], dim=1)
        

    def __len__(self):
        return len(self.targets)
    

    def __getitem__(self, index):
        datum, target = self.data[index], self.targets[index].int()
        return datum, target


def load_pneumonia_features(which: str):
    if which == 'train':
        return _IndexedPneumoniaFeatures(train=True, indexed=True)
    elif which == 'test':
        return _IndexedPneumoniaFeatures(train=False, indexed=False)
    else:
        raise ValueError("Can only choose between 'train' and 'test'.")
    