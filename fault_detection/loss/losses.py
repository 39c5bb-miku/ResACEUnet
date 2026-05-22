import torch.nn as nn
from monai.losses.dice import DiceLoss, DiceCELoss, DiceFocalLoss


class CrossEntropy(nn.Module):
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
class Dice(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = DiceLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


###########################################################################
class DiceCE(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = DiceCELoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


###########################################################################
class DiceFocal(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = DiceFocalLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


###########################################################################
def build_loss_fn(config):
    if config.loss.name == "ce":
        return CrossEntropy()

    elif config.loss.name == "bce":
        return BinaryCrossEntropyWithLogits()

    elif config.loss.name == "dice":
        return Dice()

    elif config.loss.name == "dice_ce":
        return DiceCE()
    elif config.loss.name == "dice_focal":
        return DiceFocal()
    else:
        raise ValueError("must be cross entropy or soft dice loss for now!")
