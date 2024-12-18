from monai.optimizers.lr_scheduler import WarmupCosineSchedule,LinearLR,ExponentialLR

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.train.epochs * n_iter_per_epoch)
    warmup_steps = int(config.lr_scheduler.warmup_epochs * n_iter_per_epoch)

    lr_scheduler = None
    if config.lr_scheduler.name == 'cosine':
        lr_scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            t_total=num_steps,
            end_lr=config.lr_scheduler.min_lr,
            warmup_multiplier=0.1
        )
    elif config.lr_scheduler.name == 'linear':
        lr_scheduler = LinearLR(
            optimizer=optimizer,
            end_lr=config.lr_scheduler.min_lr,
            num_iter=num_steps,
        )
    elif config.lr_scheduler.name == 'exponential':
        lr_scheduler = ExponentialLR(
            optimizer=optimizer,
            end_lr=config.lr_scheduler.min_lr,
            num_iter=num_steps,
        )
    elif config.lr_scheduler.name == 'None':
        lr_scheduler = None
    return lr_scheduler

