import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from typing import Any, Tuple
from typing_extensions import override


CIFAR_PARAMS = {
    'dataset_name': 'CIFAR10',
    'n_data': 50_000,
    'shape': (3, 32, 32),
    'mean': (0.4914, 0.4822, 0.4465),
    'std': (0.2023, 0.1994, 0.2010),
    'root': 'cifar',
}


CIFAR_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR_PARAMS['mean'], std=CIFAR_PARAMS['std'])])


class _IndexedCIFAR(datasets.CIFAR10):
    def __init__(
            self, train=True, indexed=True
    ):
        super().__init__(root=f"datasets/{CIFAR_PARAMS['root']}", 
                         train=train, 
                         transform=CIFAR_TRANSFORM, 
                         target_transform=None, 
                         download=True)
        
        if not train:
            self.data = torch.from_numpy(self.data).float()
            self.targets = torch.from_numpy(np.array(self.targets)).int()
        else:
            self.data = torch.from_numpy(self.data).float()
            targets = torch.from_numpy(np.array(self.targets)).int()
            targets_labelled = torch.cat((targets.reshape(-1, 1), torch.arange(len(targets)).reshape(-1, 1)), dim=1)
            self.targets = targets_labelled

    @override
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index].int()
        img = Image.fromarray(img.numpy().astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

def load_cifar(which: str):
    """
    Loads the CIFAR-10 dataset.

    Args:
        which (str): `train` or `test`.

    Returns:
        The required version of CIFAR-10 dataset.
    """
    if which == 'train':
        return _IndexedCIFAR(train=True, indexed=True)
    elif which == 'test':
        return _IndexedCIFAR(train=False, indexed=False)
    else:
        raise ValueError("Can only choose between 'train' and 'test'.")
    