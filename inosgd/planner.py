# TO CLEAN UP

import torch
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from scipy import optimize
from tqdm import tqdm

from inosgd.ipp import IPP
from inosgd.privacy.privacy_accounting import get_sampling_rate, DEFAULT_ALPHAS, get_sampling_rate_new


class IPPPlanner:
    def __init__(self, 
                 n_iters: int,
                 dataset_sizes: torch.Tensor,
                 epsilons: torch.Tensor,
                 deltas: torch.Tensor,
                 assignment: torch.Tensor):
        self.n_iters = n_iters
        self.dataset_sizes = dataset_sizes
        self.n_data = int(torch.sum(self.dataset_sizes))
        self.n_owners = len(self.dataset_sizes)
        self.epsilons = epsilons
        self.deltas = deltas
        self.assignment = assignment


    def _find_noise_multiplier(self, batch_size: float):
        sigma_low, sigma_high = 1e-2, 10000.0

        def bisect_objective(sigma):
            sampling_rates = torch.full((self.n_owners,), 1).float()
            for o in range(self.n_owners):
                target_epsilon = float(self.epsilons[o])
                target_delta = float(self.deltas[o])
                sampling_rates[o] = get_sampling_rate(target_epsilon, target_delta, sigma, self.n_iters)
            return torch.sum(self.dataset_sizes * sampling_rates).item() - batch_size
        
        result = optimize.bisect(bisect_objective, sigma_low, sigma_high)
        #print("Find:" + str(result))
        return result


    def plan_sampling(self, batch_size, clipping_threshold):
        noise_multiplier = self._find_noise_multiplier(batch_size)
        sampling_rates = torch.full((self.n_owners,), 1).float()
        for o in range(self.n_owners):
            target_epsilon = float(self.epsilons[o])
            target_delta = float(self.deltas[o])
            sampling_rates[o] = get_sampling_rate(target_epsilon, target_delta, noise_multiplier, self.n_iters)
        
        sampling_rates = sampling_rates.repeat(self.n_iters, 1)
        noise_scales = torch.full((self.n_iters, 1), noise_multiplier * clipping_threshold)
        clipping_thresholds = torch.full((self.n_iters, self.n_owners), clipping_threshold)
        return IPP(sampling_rates, clipping_thresholds, noise_scales, self.assignment, self.dataset_sizes)
        

    

    def plan_sampling_min(self, batch_size, clipping_threshold): # faster
        most_private_owner_ind = torch.argmin(self.epsilons)
        print(f"Planning Owner {most_private_owner_ind}...")
        noise_multiplier = get_noise_multiplier(target_epsilon=self.epsilons[most_private_owner_ind].item(),
                                                target_delta=self.deltas[most_private_owner_ind].item(),
                                                sample_rate=batch_size / self.n_data,
                                                steps=self.n_iters,
                                                alphas=DEFAULT_ALPHAS,
                                                epsilon_tolerance=1e-5)
        noise_scale = clipping_threshold * noise_multiplier
        sampling_rates = torch.full((self.n_owners,), 1).float()
        for o in range(self.n_owners):
            if o == most_private_owner_ind:
                sampling_rates[o] = batch_size / self.n_data
                continue
            print(f"Planning Owner {o}...")
            target_epsilon = float(self.epsilons[o])
            target_delta = float(self.deltas[o])
            sampling_rates[o] = get_sampling_rate_new(target_epsilon, target_delta, noise_multiplier, self.n_iters)
        
        sampling_rates = sampling_rates.repeat(self.n_iters, 1)
        noise_scales = torch.full((self.n_iters, 1), noise_scale)
        clipping_thresholds = torch.full((self.n_iters, self.n_owners), clipping_threshold)
        return IPP(sampling_rates, clipping_thresholds, noise_scales, self.assignment, self.dataset_sizes)
    

    def plan_sample(self, batch_size, clipping_threshold, bmin=1, bmax=None):
        if not bmax:
            bmax = batch_size
        def bisect_objective(b):
            print(f"Trying min batch size {b}...")
            bactual = self.plan_sampling_min(b, clipping_threshold).get_batch_sizes()[0].item()
            print(f"Total batch size is {bactual}.")
            if abs(bactual - batch_size) < 0.05:
                return 0
            else:
                return bactual - batch_size
        
        result = optimize.bisect(bisect_objective, bmin, bmax)
        return self.plan_sampling_min(result, clipping_threshold)
    

    def plan_stretching_min(self, batch_size, clipping_threshold):
        sampling_rate = batch_size / self.n_data
        sampling_rates = torch.full((self.n_iters, self.n_owners), sampling_rate)
        most_private_owner_ind = torch.argmin(self.epsilons)
        noise_multiplier = get_noise_multiplier(target_epsilon=self.epsilons[most_private_owner_ind].item(),
                                                target_delta=self.deltas[most_private_owner_ind].item(),
                                                sample_rate=sampling_rate,
                                                steps=self.n_iters,
                                                alphas=DEFAULT_ALPHAS,
                                                epsilon_tolerance=1e-5)
        noise_scale = clipping_threshold * noise_multiplier
        noise_scales = torch.full((self.n_iters, 1), noise_scale)
        clipping_thresholds = torch.full((self.n_owners,), clipping_threshold).float()
        for o in range(self.n_owners):
            print(f"Owner {o} planned.")
            if o == most_private_owner_ind:
                continue
            else:
                noise_multiplier = get_noise_multiplier(target_epsilon=self.epsilons[o].item(),
                                                        target_delta=self.deltas[o].item(),
                                                        sample_rate=sampling_rate,
                                                        steps=self.n_iters,
                                                        alphas=DEFAULT_ALPHAS,
                                                        epsilon_tolerance=1e-5)
                clipping_thresholds[o] = noise_scale / noise_multiplier

        clipping_thresholds = clipping_thresholds.repeat((self.n_iters, 1))
        return IPP(sampling_rates, clipping_thresholds, noise_scales, self.assignment, self.dataset_sizes)


    def plan_stretching_according(self, batch_size, clipping_threshold, privacy_budget=0.5, acc_delta=1e-5):
        sampling_rate = batch_size / self.n_data
        sampling_rates = torch.full((self.n_iters, self.n_owners), sampling_rate)
        noise_multiplier = get_noise_multiplier(target_epsilon=privacy_budget,
                                                target_delta=acc_delta,
                                                sample_rate=sampling_rate,
                                                steps=self.n_iters,
                                                alphas=DEFAULT_ALPHAS,
                                                epsilon_tolerance=1e-5)
        noise_scale = clipping_threshold * noise_multiplier
        noise_scales = torch.full((self.n_iters, 1), noise_scale)
        clipping_thresholds = torch.full((self.n_owners,), clipping_threshold).float()
        for o in tqdm(range(self.n_owners)):
            noise_multiplier = get_noise_multiplier(target_epsilon=self.epsilons[o].item(),
                                                    target_delta=self.deltas[o].item(),
                                                    sample_rate=sampling_rate,
                                                    steps=self.n_iters,
                                                    alphas=DEFAULT_ALPHAS,
                                                    epsilon_tolerance=1e-5)
            clipping_thresholds[o] = noise_scale / noise_multiplier

        clipping_thresholds = clipping_thresholds.repeat((self.n_iters, 1))
        return IPP(sampling_rates, clipping_thresholds, noise_scales, self.assignment, self.dataset_sizes)



    