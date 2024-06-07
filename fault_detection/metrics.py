import torch
import torch.nn as nn
from torch.nn import functional as F

class AverageMeter:
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def compute_iou(im1, im2):
	device = im1.device
	im1 = torch.where(im1 > 0.5, torch.tensor(1,device=device), torch.tensor(0,device=device))
	TP = ((im1 == 1) & (im2 == 1)).sum().item()
	FP = ((im1 == 1) & (im2 == 0)).sum().item()
	FN = ((im1 == 0) & (im2 == 1)).sum().item()
	iou = TP / (FP + TP + FN + 1E-8)

	return iou


def compute_dice(im1, im2):
	device = im1.device
	im1 = torch.where(im1 > 0.5, torch.tensor(1,device=device), torch.tensor(0,device=device))
	TP = ((im1 == 1) & (im2 == 1)).sum().item()
	FP = ((im1 == 1) & (im2 == 0)).sum().item()
	FN = ((im1 == 0) & (im2 == 1)).sum().item()
	dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
	return dice


def compute_acc(im1, im2):
	device = im1.device
	im1 = torch.where(im1 > 0.5, torch.tensor(1,device=device), torch.tensor(0,device=device))
	TP = ((im1 == 1) & (im2 == 1)).sum().item()
	FP = ((im1 == 1) & (im2 == 0)).sum().item()
	TN = ((im1 == 0) & (im2 == 0)).sum().item()
	FN = ((im1 == 0) & (im2 == 1)).sum().item()
	acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
	return acc


def compute_prec(im1, im2):
	device = im1.device
	im1 = torch.where(im1 > 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device))
	TP = ((im1 == 1) & (im2 == 1)).sum().item()
	FP = ((im1 == 1) & (im2 == 0)).sum().item()
	prec = TP / (TP + FP + 1e-8)
	return prec


def compute_recall(im1, im2):
	device = im1.device
	im1 = torch.where(im1 > 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device))
	TP = ((im1 == 1) & (im2 == 1)).sum().item()
	FN = ((im1 == 0) & (im2 == 1)).sum().item()
	recall = TP / (TP + FN + 1e-8)
	return recall


def compute_f1(im1, im2):
	prec = compute_prec(im1, im2)
	recall = compute_recall(im1, im2)
	f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
	return f1


def dice_coef(y_pred, y_true, smooth_nr, smooth_dr):
	batch_size = y_true.size(0)
	results = 0
	for i in range(batch_size):
		y_true_s, y_pred_s = y_true[i, 0, :, :], y_pred[i, 0, :, :]
		y_true_f = torch.flatten(y_true_s)
		y_pred_f = torch.flatten(y_pred_s)
		intersection = torch.sum(y_true_f * y_pred_f)
		results += (2. * intersection + smooth_nr) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth_dr)
	return results / batch_size


class dice_coef_loss(nn.Module):
	def __init__(self, squared_pred=False, smooth_nr=1e-5, smooth_dr=1e-5):  ### nr=0  dr=1e-6
		super(dice_coef_loss, self).__init__()
		self.smooth_nr = float(smooth_nr)
		self.smooth_dr = float(smooth_dr)
		self.squared_pred = squared_pred

	def forward(self, y_pred, y_true):
		if self.squared_pred:
			y_pred = torch.pow(y_pred, 2)
			y_true = torch.pow(y_true, 2)
		return 1. - dice_coef(y_pred, y_true, self.smooth_nr, self.smooth_dr)


class bce_loss(nn.Module):
	def __init__(self, reduction='mean', smoothing=0.05):
		super(bce_dice_loss, self).__init__()
		self.smoothing = smoothing
		self.reduction = reduction
		self.weight = None

	@staticmethod
	def _smooth(targets: torch.Tensor, smoothing=0.0):
		assert 0 <= smoothing < 1
		with torch.no_grad():
			targets = targets * (1.0 - smoothing) + 0.5 * smoothing
		return targets

	def forward(self, y_pred, y_true):
		y_true = bce_loss._smooth(y_true, self.smoothing)
		return F.binary_cross_entropy(y_pred, y_true, weight=self.weight, reduction=self.reduction)


class bce_dice_loss(nn.Module):
	def __init__(self, bce_weight=0.5, smooth_nr=1e-5, smooth_dr=1e-5, reduction='mean', smoothing=0.05):
		super(bce_dice_loss, self).__init__()
		self.bce_weight = bce_weight
		self.smooth_nr = float(smooth_nr)
		self.smooth_dr = float(smooth_dr)
		self.reduction = reduction
		self.smoothing = smoothing
		self.weight = None

	@staticmethod
	def _smooth(targets: torch.Tensor, smoothing=0.0):
		assert 0 <= smoothing < 1
		with torch.no_grad():
			targets = targets * (1.0 - smoothing) + 0.5 * smoothing
		return targets

	def forward(self, y_pred, y_true):
		y_true = bce_dice_loss._smooth(y_true, self.smoothing)
		return F.binary_cross_entropy_with_logits(y_pred, y_true, weight=self.weight,
									  reduction=self.reduction) * self.bce_weight + (
				1. - dice_coef(y_pred, y_true, self.smooth_nr, self.smooth_dr)) * (1 - self.bce_weight)