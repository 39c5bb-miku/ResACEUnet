# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from monai import transforms

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.train.epochs * n_iter_per_epoch)#迭代步数
    warmup_steps = int(config.lr_scheduler.warmup_epochs * n_iter_per_epoch)#预热步数
    decay_steps = int(config.lr_scheduler.decay_epochs * n_iter_per_epoch)#衰减步数

    lr_scheduler = None
    if config.lr_scheduler.name == 'cosine':#余弦衰减
        lr_scheduler = CosineLRScheduler(
            optimizer,#优化器
            t_initial=num_steps,#迭代步数
            t_mul=1.,
            lr_min=config.lr_scheduler.min_lr,#最小学习率
            warmup_lr_init=config.lr_scheduler.warmup_lr,#预热学习率
            warmup_t=warmup_steps,#预热步数
            cycle_limit=1,#循环次数
            t_in_epochs=False,#是否以epoch为单位
        )
    elif config.lr_scheduler.name == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.lr_scheduler.warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.lr_scheduler.name == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.train.lr_scheduler.decay_rate,
            warmup_lr_init=config.lr_scheduler.warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.lr_scheduler.name == 'None':
        lr_scheduler = None
    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                optimizer: torch.optim.Optimizer,
                t_initial: int,
                lr_min_rate: float,
                warmup_t=0,
                warmup_lr_init=0.,
                t_in_epochs=True,
                noise_range_t=None,
                noise_pct=0.67,
                noise_std=1.0,
                noise_seed=42,
                initialize=True,
                ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
