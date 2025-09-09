from typing import Any, List, Union
from tqdm import tqdm

import gc
import numpy as np
import torch
from opacus.grad_sample import AbstractGradSampleModule
from scipy.integrate import cumtrapz
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

from inosgd.ipp import IPP
from inosgd.dataloader import INODataLoader
from inosgd.optimizer import INOOptimizer
from inosgd.evaluate import evaluate


def ino_train(train_loader: INODataLoader,
              model: AbstractGradSampleModule,
              optimizer: INOOptimizer,
              test_loader: DataLoader,
              ipp: IPP,
              bif: Any,
              device: Any,
              evaluate_fn=evaluate,
              adaptive_threshold=0,
              max_n_steps=None,
              fast_integrate=None):
    if not max_n_steps:
        max_n_steps = ipp.get_n_iterations()

    n_epochs = n_steps = 0
    results = []

    performances = evaluate_fn(model, test_loader, device)
    performances['train_loss'] = performances['eval_loss'] # random
    results.append(performances)

    F_grid = None
    if fast_integrate is not None:
        pts = np.linspace(0, fast_integrate[0], fast_integrate[0] * fast_integrate[1] + 1) # precision
        f_grid = bif(pts)
        F_grid = torch.from_numpy(cumtrapz(f_grid, pts, initial=0)).float()

    pbar = tqdm(total=max_n_steps)
    while n_steps < max_n_steps:
        n_steps, performances = train_one_epoch(train_loader,
                                              model,
                                              optimizer,
                                              ipp,
                                              bif,
                                              n_steps,
                                              max_n_steps,
                                              device,
                                              adaptive_threshold,
                                              evaluate_fn=evaluate_fn,
                                              test_loader=test_loader,
                                              results=results,
                                              fast_integrate=fast_integrate,
                                              F_grid=F_grid)

        n_epochs += 1
        print(f"Epoch {n_epochs}: {performances}")
        pbar.update(n_steps - pbar.n)

    pbar.close()
    return results


def train_one_epoch(train_loader: INODataLoader,
                    model: AbstractGradSampleModule,
                    optimizer: INOOptimizer,
                    ipp: IPP,
                    bif: None,
                    n_steps: int,
                    max_n_steps: int,
                    device: Any,
                    adaptive_threshold: float,
                    evaluate_fn: Any,
                    test_loader: DataLoader,
                    results,
                    fast_integrate,
                    F_grid):
    model.train()
    criterion = F.cross_entropy
    losses = []

    for _, (data, target) in enumerate(train_loader):
        
        indices = target[:, 1]
        target = target[:, 0]

        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        
        output = model(data)
        batch_size = ipp.get_batch_sizes()[n_steps].item()
        clipping_thresholds = ipp.get_per_sample_clipping_thresholds()[n_steps]
        noise_scale = ipp.noise_scales[n_steps].item()
        
        batch_clipping_thresholds = clipping_thresholds[indices].to(device)
        loss = criterion(output, target, reduction='none')
        loss_clone = loss.clone().detach()
        if fast_integrate is not None:
            gradient_multipliers = _get_gradient_multipliers_fast(loss_clone, batch_clipping_thresholds, fast_integrate, F_grid)
        else:
            gradient_multipliers = _get_gradient_multipliers(loss_clone, batch_clipping_thresholds, bif)

        gradient_multipliers = gradient_multipliers.to(device)
        
        del loss_clone

        loss = torch.mean(loss)

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()

        optimizer.step(batch_size,
                       batch_clipping_thresholds,
                       noise_scale, 
                       gradient_multipliers=gradient_multipliers,
                       adaptive_threshold=adaptive_threshold)

        n_steps += 1
        if n_steps % 100 == 0:
            print(n_steps)
        if n_steps == max_n_steps:
            break
    
    performances = evaluate_fn(model, test_loader, device)
    performances['train_loss'] = np.mean(losses)
    results.append(performances)
    model.train()
    return n_steps, performances


def _get_gradient_multipliers(batch_losses: List[float], 
                              batch_clipping_thresholds: List[float], 
                              bif) -> List[float]:
    """
    Modifies the batch per-sample clipping thresholds based on the INO-SGD algorithm.

    Args:
        batch_losses: Individual losses corresponding to each datum in the batch.
        batch_clipping_thresholds: Clipping threshold associated with each datum in the batch.
        bif: Specifies the BIF.

    Returns:
        List[float]: The modified per-sample clipping thresholds.
    """
    order = torch.argsort(batch_losses).cpu()
    accumulated_threshold = 0
    gradient_multipliers = torch.zeros((len(batch_losses),))
    for o in range(len(batch_losses)):
        curr_threshold = batch_clipping_thresholds[order[o]].cpu()
        pts = torch.linspace(accumulated_threshold, accumulated_threshold + curr_threshold, steps=100)
        pts_values = torch.Tensor(bif(pts))
        gradient_multipliers[order[o]] = torch.trapezoid(pts_values, x=pts) / curr_threshold
        accumulated_threshold += curr_threshold

    return gradient_multipliers


def _get_gradient_multipliers_fast(batch_losses: List[float],
                                   batch_clipping_thresholds: List[float],
                                   fast_integrate,
                                   F_grid) -> List[float]:
    order = torch.argsort(batch_losses).cpu()
    accumulated_threshold = 0
    gradient_multipliers = torch.zeros((len(batch_losses),))
    ordered_thresholds = batch_clipping_thresholds[order].cpu()
    endpoints = torch.round(torch.cumsum(ordered_thresholds, dim=0) * fast_integrate[1]).int()
    startpoints = torch.zeros_like(endpoints).int()
    startpoints[1:] = endpoints[:-1]
    gradient_multipliers = F_grid[endpoints] - F_grid[startpoints]
    gradient_multipliers /= ordered_thresholds
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    gradient_multipliers = gradient_multipliers[inv_order]

    return gradient_multipliers
