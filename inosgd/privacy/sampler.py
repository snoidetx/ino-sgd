from typing import List

import numpy as np
import torch
from torch.utils.data import Sampler


class IDPSampler(Sampler[List[int]]):
    r"""
    Each sample is selected with a probability equal to its individualized sample rate.
    The sampler generates ``steps`` number of batches.
    """

    def __init__(
        self, *, n_samples: int, per_sample_sampling_rates: List[float], generator=None, n_steps=None, max_batch_size=None, min_per_group=None, targets=None
    ):
        r"""
        Args:
            num_samples: number of samples to draw.
            sample_rate: probability used in sampling.
            generator: Generator used in sampling.
            steps: Number of steps (iterations of the Sampler)
        """
        self.num_samples = n_samples
        self.sample_rates = per_sample_sampling_rates
        self.generator = generator
        self.max_batch_size = max_batch_size
        self.targets = targets
        self.min_per_group = min_per_group

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )
        
        if self.num_samples != len(self.sample_rates):
            raise ValueError(
                "Number of data {} is different from number of sample rates {}".format(self.num_samples, len(self.sample_rates))
            )

        if n_steps is not None:
            self.steps = n_steps
        else:
            self.steps = int(torch.sum(self.sample_rates[0])) # problematic


    def __len__(self):
        return self.steps

    def __iter__(self):
        num_batches = self.steps
        while num_batches > 0:
            while True:
                mask = (
                    torch.rand(self.num_samples, generator=self.generator)
                    < self.sample_rates
                )
                
                indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
               
                if not self.max_batch_size and not self.min_per_group:
                    break
                elif self.max_batch_size != None and len(indices) > self.max_batch_size: # for gpu reasons
                    continue
                elif self.min_per_group != None and not (torch.bincount(self.targets[indices]) >= self.min_per_group).all():
                    continue
                else:
                    break
          
            yield indices

            num_batches -= 1
