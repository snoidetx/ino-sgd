import torch


class IPP:
    r"""
    An individualized privacy profile (IPP) object stores the per-iteration per-owner sampling_rates, clipping_thresholds and noise_scales.

    Args:
        sampling_rates: `n_iterations` :math:`\times` `n_owners` tensor.
        clipping_thresholds: `n_iterations` :math:`\times` `n_owners` tensor.
        noise_scales: `n_iterations` tensor.
    """
    def __init__(self, sampling_rates, clipping_thresholds, noise_scales, assignment, dataset_sizes):
        self.sampling_rates = sampling_rates
        self.clipping_thresholds = clipping_thresholds
        self.noise_scales = noise_scales
        self.assignment = assignment
        self.dataset_sizes = dataset_sizes


    def get_per_sample_sampling_rates(self):
        if not hasattr(self, "per_sample_sampling_rates"):
            self.per_sample_sampling_rates = self.sampling_rates[:, self.assignment]
        return self.per_sample_sampling_rates


    def get_per_sample_clipping_thresholds(self):
        if not hasattr(self, "per_sample_clipping_thresholds"):
            self.per_sample_clipping_thresholds = self.clipping_thresholds[:, self.assignment]
        return self.per_sample_clipping_thresholds
    

    def get_batch_sizes(self):
        if not hasattr(self, "batch_sizes"):
            self.batch_sizes = torch.sum(self.sampling_rates * self.dataset_sizes, dim=1)
        return self.batch_sizes
    

    def get_n_iterations(self):
        return len(self.noise_scales)
