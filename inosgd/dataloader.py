from typing import List

import torch
from opacus.data_loader import dtype_safe, shape_safe, wrap_collate_with_empty
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate

from inosgd.ipp import IPP
from inosgd.privacy.sampler import IDPSampler


class INODataLoader(DataLoader):
    """
    Modifies the original `opacus.data_loader.DPDataLoader` for INO-SGD.

    Args:
        dataset: the dataset to be loaded.
        per_sample_sampling_rates: The sampling rate of each sample in `dataset`.
    """
    def __init__(self, dataset: Dataset, per_sample_sampling_rates: torch.Tensor, collate_fn=None, max_batch_size=None, min_per_group=None):
        n_data = len(dataset)
        batch_size = torch.sum(per_sample_sampling_rates).item()
        n_steps = int(n_data / batch_size)
        batch_sampler = IDPSampler(
            n_samples=len(dataset),
            per_sample_sampling_rates=per_sample_sampling_rates,
            n_steps=n_steps,
            max_batch_size=max_batch_size,
            min_per_group=min_per_group,
            targets=dataset.targets[:, 0]
        )

        if collate_fn is None:
            collate_fn = default_collate
        sample_empty_shapes = [(0, *shape_safe(x)) for x in dataset[0]]
        dtypes = [dtype_safe(x) for x in dataset[0]]

        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=wrap_collate_with_empty(
                collate_fn=collate_fn,
                sample_empty_shapes=sample_empty_shapes,
                dtypes=dtypes,
            ),
            generator=None)
        
    
    @classmethod
    def from_data_loader(cls, data_loader: DataLoader, ipp: IPP, **kwargs):
        """
        Privatize the given `data_loader` according to `ipp`.

        Args:
            data_loader: The `DataLoader` to be privatized.
            ipp: The `IPP` profile to follow.
        """
        per_sample_sampling_rates = ipp.get_per_sample_sampling_rates()[0]
        max_batch_size = None if "max_batch_size" not in kwargs else kwargs["max_batch_size"]
        min_per_group = None if "min_per_group" not in kwargs else kwargs["min_per_group"]

        return cls(
            dataset=data_loader.dataset,
            per_sample_sampling_rates=per_sample_sampling_rates,
            collate_fn=data_loader.collate_fn,
            max_batch_size=max_batch_size,
            min_per_group=min_per_group
        )
