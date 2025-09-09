from typing import Tuple

from opacus.accountants import IAccountant, create_accountant
from opacus.grad_sample import AbstractGradSampleModule
from torch import nn, optim
from torch.utils.data import DataLoader

from inosgd.ipp import IPP
from inosgd.dataloader import INODataLoader
from inosgd.model import INOModel
from inosgd.optimizer import INOOptimizer


class INOPrivacyEngine:
    """
    Modifies the original `opacus.PrivacyEngine` for INO-SGD.

    Attributes:
        accountant (IAccountant): The privacy accountant to be used.
    """
    def __init__(self):
        self.accountant = create_accountant(mechanism='rdp')

    
    def make_private(self, *, 
                     dataloader: DataLoader,
                     model: nn.Module,
                     optimizer: optim.Optimizer,
                     ipp: IPP, **kwargs) -> Tuple[INODataLoader, AbstractGradSampleModule, INOOptimizer]:
        """
        Privatizes the given `data_loader`, `model` and `optimizer` according to `ipp`.
        """
        ino_dataloader = INODataLoader.from_data_loader(dataloader, ipp, **kwargs)
        ino_model = INOModel(model)
        ino_optimizer = INOOptimizer(optimizer)
        return (ino_dataloader, ino_model, ino_optimizer)
    