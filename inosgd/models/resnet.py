from torchvision import models
from opacus.validators import ModuleValidator


class CifarResNet:
    def __new__(cls):
        model = models.resnet18(num_classes=10)
        model = ModuleValidator.fix(model)
        return model
  