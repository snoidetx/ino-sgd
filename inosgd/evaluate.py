from typing import Any, List

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models


def evaluate(model: nn.Module, test_loader: DataLoader, device: Any) -> float:
    """
    Evaluates the performance of `model` using data from `test_loader`. In particular, the evaluation losses, accuracies and per-class recalls will be returned.

    Args:
        model: The `nn.Module` model to be evaluated.
        test_loader: Loads the test/validation set.
        device: CPU or GPU.

    Returns:
        dic: A dictionary containing the performances.
    """
    model.eval()
    n_correct = 0
    losses = []
    criterion = F.cross_entropy
    labels = torch.unique(test_loader.dataset.targets)
    per_class_n_data = {l.item(): 0 for l in labels}
    per_class_n_correct = {l.item(): 0 for l in labels}
    per_class_avg_losses = {l.item(): 0 for l in labels}
    with torch.no_grad():
        for (data, target) in test_loader:
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            output = model(data)
            l = criterion(output, target, reduction='none')
            losses.append(l.mean().item())
            pred = output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            for i in range(len(target)):
                per_class_n_data[target[i].item()] += 1
                per_class_avg_losses[target[i].item()] += l[i].item()

                if target[i].item() == pred[i].item():
                    per_class_n_correct[target[i].item()] += 1

    mean_loss = float(np.mean(losses))
    accuracy = 100. * n_correct / len(test_loader.dataset)
    per_class_losses = [per_class_avg_losses[l.item()] / per_class_n_data[l.item()] for l in labels]
    per_class_recalls = [100. * per_class_n_correct[l.item()] / per_class_n_data[l.item()] for l in labels]
    balanced_accuracy = np.mean(per_class_recalls)

    return {'eval_loss': mean_loss, 
            'accuracy': accuracy, 
            'per_class_recalls': per_class_recalls, 
            'per_class_losses': per_class_losses, 
            'balanced_accuracy': balanced_accuracy}
