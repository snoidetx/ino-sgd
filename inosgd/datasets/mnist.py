import torch
from PIL import Image
from torchvision import datasets, transforms
from typing import Any, Tuple
from typing_extensions import override


MNIST_PARAMS = {
    'dataset_name': 'MNIST',
    'n_data': 60_000,
    'shape': (1, 28, 28),
    'mean': (0.1307,),
    'std': (0.3081,),
    'root': '',
}

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MNIST_PARAMS['mean'], std=MNIST_PARAMS['std'])])

MNIST_PER_CLASS_MAX_COUNT = {2:4586, 5:4586, 9:4586, 
                             3:5800, 7:5800, 8:5800, 
                             0:2327, 1:2327, 4:2327, 6:2327}


class _IndexedMNIST(datasets.MNIST):
    def __init__(
            self, train=True, indexed=True, imbalanced=False
    ):
        super().__init__(root=f"datasets/{MNIST_PARAMS['root']}", 
                         train=train, 
                         transform=MNIST_TRANSFORM, 
                         target_transform=None, 
                         download=True)
        
        if not train:
            pass
        else:
            if imbalanced:
                selected_indices = []
                per_class_counts = {c:0 for c in MNIST_PER_CLASS_MAX_COUNT}
                for i, target in enumerate(self.targets):
                    target = target.item()
                    if per_class_counts[target] < MNIST_PER_CLASS_MAX_COUNT[target]:
                        selected_indices.append(i)
                        per_class_counts[target] += 1
                    else:
                        continue

                selected_indices = torch.Tensor(selected_indices).int()
                self.data = self.data[selected_indices]
                self.targets = self.targets[selected_indices]

            targets = self.targets
            targets_labelled = torch.cat((targets.reshape(-1, 1), torch.arange(len(targets)).reshape(-1, 1)), dim=1)
            self.targets = targets_labelled


    @override
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index].int()
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

def load_mnist(which: str):
    """
    Loads the MNIST dataset.

    Args:
        which (str): `train` or `test`.

    Returns:
        The required version of SVHN dataset.
    """
    if which == 'train':
        return _IndexedMNIST(train=True, indexed=True)
    elif which == 'test':
        return _IndexedMNIST(train=False, indexed=False)
    else:
        raise ValueError("Can only choose between 'train' and 'test'.")
    