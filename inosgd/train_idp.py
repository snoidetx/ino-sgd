from typing import Any, List, Union
from tqdm import tqdm

import gc
import numpy as np
import torch
from opacus.grad_sample import AbstractGradSampleModule
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

from inosgd.ipp import IPP
from inosgd.dataloader import INODataLoader
from inosgd.optimizer import INOOptimizer
from inosgd.evaluate import evaluate


def idp_train(train_loader: INODataLoader,
              model: AbstractGradSampleModule,
              optimizer: INOOptimizer,
              test_loader: DataLoader,
              ipp: IPP,
              device: Any,
              adaptive_threshold=0,
              max_n_steps=None,
              evaluate_fn=evaluate):
    if not max_n_steps:
        max_n_steps = ipp.get_n_iterations()

    n_epochs = n_steps = 0
    results = []

    performances = evaluate_fn(model, test_loader, device)
    performances['train_loss'] = performances['eval_loss'] # random
    results.append(performances)

    pbar = tqdm(total=max_n_steps)
    while n_steps < max_n_steps:
        n_steps, performances = train_one_epoch(train_loader,
                                              model,
                                              optimizer,
                                              ipp,
                                              n_steps,
                                              max_n_steps,
                                              device,
                                              adaptive_threshold,
                                              evaluate_fn,
                                              test_loader,
                                              results)

        n_epochs += 1
        print(f"Epoch {n_epochs}: {performances}")
        pbar.update(n_steps - pbar.n)

    pbar.close()
    return results


def train_one_epoch(train_loader: INODataLoader,
                    model: AbstractGradSampleModule,
                    optimizer: INOOptimizer,
                    ipp: IPP,
                    n_steps: int,
                    max_n_steps: int,
                    device: Any,
                    adaptive_threshold: float,
                    evaluate_fn: Any,
                    test_loader: DataLoader,
                    results):
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
        loss = criterion(output, target, reduction='mean')
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()

        optimizer.step(batch_size,
                       batch_clipping_thresholds,
                       noise_scale, 
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
