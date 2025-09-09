from typing import Callable, List, Optional
from typing_extensions import override

import torch
from torch import optim
from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _generate_noise, _mark_as_processed
from opt_einsum import contract


class INOOptimizer(DPOptimizer):
    """
    Modifies the original `opacus.optimizers.DPOptimizer` for INO-SGD.
    """
    DEFAULT_NOISE_MULTIPLIER = 0
    DEFAULT_CLIPPING_THRESHOLD = 1
    DEFAULT_EXPECTED_BATCH_SIZE = 1


    def __init__(self, optimizer: optim.Optimizer):
        super().__init__(optimizer, 
                         noise_multiplier=self.DEFAULT_NOISE_MULTIPLIER,
                         max_grad_norm=self.DEFAULT_CLIPPING_THRESHOLD,
                         expected_batch_size=self.DEFAULT_EXPECTED_BATCH_SIZE)
        #self.recorder = []
        

    @override
    def clip_and_accumulate(self, clipping_threshold, gradient_multipliers, adaptive_threshold=0):
        if len(self.grad_samples[0]) == 0:
            per_sample_clip_factor = torch.zeros(
                (0,), device=self.grad_samples[0].device
            )
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            #print(#torch.bincount((per_sample_norms > 1).int())[1].item(),
            #      torch.bincount((per_sample_norms > .5).int(), minlength=2)[1].item(),
            #      torch.bincount((per_sample_norms <= .5).int(), minlength=2)[1].item())
                  #torch.bincount((per_sample_norms > .2).int())[1].item(),
                  #torch.bincount((per_sample_norms > .1).int())[1].item())
            if adaptive_threshold == 0: # abadi's clipping
                per_sample_clip_factor = (
                    clipping_threshold / (per_sample_norms + 1e-6)
                ).clamp(max=1.0)
            else: # adaptive clipping
                per_sample_clip_factor = (
                    clipping_threshold / (per_sample_norms + adaptive_threshold) # / (per_sample_norms + adaptive_threshold))
                )

        if gradient_multipliers is not None:
            #print(per_sample_clip_factor.shape, gradient_multipliers.shape)
            #print(per_sample_clip_factor.device, gradient_multipliers.device)
            per_sample_clip_factor = per_sample_clip_factor * gradient_multipliers

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = contract("i,i...", per_sample_clip_factor, grad_sample)
            #if gradient_multipliers is None:
            #    grad = contract("i,i...", per_sample_clip_factor, grad_sample)
            #else:
            #    per_sample_clip_factor = per_sample_clip_factor * gradient_multipliers
                #grad = contract("i,i...", per_sample_clip_factor, grad_sample)
                #grad = contract("i,i... -> i...", gradient_multipliers, grad_sample)
                #grad = contract("i,i...", per_sample_clip_factor, grad)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    
    @override
    def add_noise(self, noise_scale):
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=noise_scale,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )

            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)
    

    @override
    def scale_grad(self, batch_size):
        if self.loss_reduction == "mean":
            for p in self.params:
                p.grad /= batch_size * self.accumulated_iterations
        

    @override
    def pre_step(self,
                 batch_size:float,
                 clipping_threshold: float,
                 noise_scale:float,
                 gradient_multipliers,
                 adaptive_threshold,
                ) -> Optional[float]:
        if self.grad_samples is None or len(self.grad_samples) == 0:
            return True

        self.clip_and_accumulate(clipping_threshold, gradient_multipliers, adaptive_threshold=adaptive_threshold)
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise(noise_scale)
        self.scale_grad(batch_size)

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True


    @override
    def step(self, 
             batch_size: float, 
             clipping_threshold: List[float], 
             noise_scale: float, 
             closure: Optional[Callable[[], float]] = None,
             gradient_multipliers=None,
             adaptive_threshold=0,
            ) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step(batch_size, clipping_threshold, noise_scale, gradient_multipliers, adaptive_threshold):
            return self.original_optimizer.step()
        else:
            return None
