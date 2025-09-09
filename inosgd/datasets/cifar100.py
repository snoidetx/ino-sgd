import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from typing import Any, Tuple
from typing_extensions import override


CIFAR100_PARAMS = {
    'dataset_name': 'CIFAR100',
    'n_data': 50_000,
    'shape': (3, 32, 32),
    'mean': (0.5071, 0.4867, 0.4408),
    'std': (0.2675, 0.2565, 0.2761),
    'root': 'cifar100',
}


CIFAR100_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR100_PARAMS['mean'], std=CIFAR100_PARAMS['std'])])


FISH_TARGETS = torch.Tensor([73, 32, 67, 91, 1])
VEHICLE_TARGETS = torch.Tensor([8, 58, 90, 13, 48])


class _IndexedCIFAR100(datasets.CIFAR100):
    def __init__(
            self, train=True, indexed=True
    ):
        super().__init__(root=f"datasets/{CIFAR100_PARAMS['root']}", 
                         train=train, 
                         transform=CIFAR100_TRANSFORM, 
                         target_transform=None, 
                         download=True)
        
        if not train:
            self.data = torch.from_numpy(self.data).float()
            self.targets = torch.from_numpy(np.array(self.targets)).int()
        else:
            self.data = torch.from_numpy(self.data).float()
            targets = torch.from_numpy(np.array(self.targets)).int()
            if indexed:
                targets_labelled = torch.cat((targets.reshape(-1, 1), torch.arange(len(targets)).reshape(-1, 1)), dim=1)
                self.targets = targets_labelled
            else:
                self.targets = targets


    @override
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index].int()
        img = Image.fromarray(img.numpy().astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

def load_cifar100(which: str):
    """
    Loads the CIFAR-100 dataset.

    Args:
        which (str): `train` or `test`.

    Returns:
        The required version of CIFAR-100 dataset.
    """
    if which == 'train':
        return _IndexedCIFAR100(train=True, indexed=True)
    elif which == 'test':
        return _IndexedCIFAR100(train=False, indexed=False)
    else:
        raise ValueError("Can only choose between 'train' and 'test'.")


def load_cifar100_fish_vehicle(which: str):
    """
    Loads the Fish and Vehicle superclasses of the CIFAR-100 dataset.

    Args:
        which (str): `train` or `test`.

    Returns: 
        The required version of CIFAR-100 dataset.
    """
    if which == 'train':
        dataset = _IndexedCIFAR100(train=True, indexed=False)
    elif which == 'test':
        dataset = _IndexedCIFAR100(train=False, indexed=False)
    else:
        raise ValueError("Can only choose between 'train' and 'test'.")
    
    kept_labels = torch.concat((FISH_TARGETS, VEHICLE_TARGETS))
    indices = torch.isin(dataset.targets[:], kept_labels).nonzero().flatten()
    dataset.data = dataset.data[indices]
    dataset.targets = dataset.targets[indices]
    for i in range(len(dataset.targets)):
        dataset.targets[i] = (kept_labels == dataset.targets[i]).nonzero(as_tuple=True)[0]
    if which == 'train':
        dataset.targets = torch.cat((dataset.targets.reshape(-1, 1), torch.arange(len(dataset.targets)).reshape(-1, 1)), dim=1)

    return dataset

