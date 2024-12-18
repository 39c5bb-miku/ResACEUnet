import torch.nn as nn
from monai import losses

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, predictions, targets):
        loss = self._loss(predictions, targets)
        return loss
###########################################################################
class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions, tragets):
        loss = self._loss(predictions, tragets)
        return loss
###########################################################################
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss
###########################################################################
class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceCELoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss
###########################################################################
class DiceFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceFocalLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss
###########################################################################
def build_loss_fn(config):
    if config.loss.name == "ce":
        return CrossEntropyLoss()

    elif config.loss.name == "bce":
        return BinaryCrossEntropyWithLogits()

    elif config.loss.name == "dice":
        return DiceLoss()

    elif config.loss.name == "dice_ce":
        return DiceCELoss()
    elif config.loss.name == "dice_focal":
        return DiceFocalLoss()
    else:
        raise ValueError("must be cross entropy or soft dice loss for now!")