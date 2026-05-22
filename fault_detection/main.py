import os
import glob
import shutil
import random
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data.distributed import (
    DistributedSampler,
)
from model.build import build_model
from data.build_data import build_loader
from data.process import tojson
from metric.metrics import *
from optimizer.lr_scheduler import build_scheduler
from optimizer.optimizer import build_optimizer
from util.distributed import (
    setup_distributed,
    reduce_tensor,
    is_main_process,
    cleanup_distributed,
    get_rank,
)
from util.logger import create_logger
from util.checkpoint import *
from util.to_torch import to_cuda
from util.config import get_config
from loss.deep_supervision import MultipleOutputLoss2, deep_supervision_scale3d
from loss.losses import build_loss_fn


def build(config):
    train_loader, val_loader = build_loader(config)
    model = build_model(config)
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    criterion = build_loss_fn(config).cuda()

    if config.model.ds:
        net_numpool = 3
        weights = torch.tensor([1 / (2**i) for i in range(net_numpool)])
        weights = weights / weights.sum()
        criterion = MultipleOutputLoss2(criterion, weights).cuda()

    return train_loader, val_loader, model, optimizer, lr_scheduler, criterion


def main(config):
    is_ddp, local_rank, world_size = setup_distributed()

    output_dir = os.path.join(config.train.output, config.model.name)
    logger = create_logger(output_dir=output_dir, dist_rank=get_rank())

    train_loader, val_loader, model, optimizer, lr_scheduler, criterion = build(config)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Creating model:{config.model.name}, number of params (M):{(n_parameters / 1.e6):.2f}"
    )

    model.cuda()
    model_without_ddp = model

    if is_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        model_without_ddp = model.module

    # if hasattr(torch, "compile"):
    #     model = torch.compile(model)

    scaler = GradScaler() if config.train.amp else None

    if is_main_process():
        wandb.init(
            project=config.model.name,
            name=config.model.name,
            config={
                "lr": config.optimizer.lr,
                "batch_size": config.data.batch_size * world_size,
                "epochs": config.train.epochs,
                "img_size": config.data.img_size,
                "loss": config.loss.name,
                "optimizer": config.optimizer.name,
                "scheduler": config.lr_scheduler.name,
                "seed": config.train.seed,
            },
        )

    if config.train.auto_resume:
        resume_file = auto_resume_helper(output_dir, logger)
        if resume_file:
            if config.model.resume:
                logger.warning(
                    f"Auto-resume changing resume file from {config.model.resume} to {resume_file}"
                )
            config.model.resume = resume_file
            logger.info(f"Auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {output_dir}, ignoring auto resume")

    if config.model.resume:
        config.train.start_epoch = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, logger
        )

    logger.info("Start training")

    min_loss = float("inf")
    max_iou, max_dice, max_acc = float("-inf"), float("-inf"), float("-inf")
    max_prec, max_recall = float("-inf"), float("-inf")
    patience, max_patience = 0, config.train.patience

    if is_main_process():
        os.makedirs(os.path.join(output_dir, "best_model"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    save_path = os.path.join(output_dir, "best_model", config.model.name)

    for epoch in range(config.train.start_epoch, config.train.epochs):
        train_loss, lr = train(
            config,
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            criterion,
            epoch,
            logger,
            scaler,
        )
        val_loss, (iou, dice, acc, prec, recall) = val(
            config, val_loader, model, criterion, logger
        )

        if ((epoch + 1) % config.train.save_freq == 0) and is_main_process():
            save_checkpoint(
                config, epoch, model_without_ddp, optimizer, lr_scheduler, logger
            )

        if val_loss <= min_loss:
            logger.info(
                f" * val_loss: {val_loss:.3f}  "
                f"iou: {iou:.3f}  "
                f"dice: {dice:.3f}  "
                f"acc: {acc:.3f}  "
                f"prec: {prec:.3f}  "
                f"recall: {recall:.3f}"
            )

            if is_main_process():
                save_path = os.path.join(output_dir, "best_model", config.model.name)
                logger.info(f"Saving best model at epoch {epoch+1}")
                torch.save(model_without_ddp.state_dict(), save_path + ".pt")
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                logger.info(
                    f"Model hasn't improved for {max_patience} epochs. Stopping training."
                )
                break

        min_loss = min(val_loss, min_loss)
        max_iou = max(iou, max_iou)
        max_dice = max(dice, max_dice)
        max_acc = max(acc, max_acc)
        max_prec = max(prec, max_prec)
        max_recall = max(recall, max_recall)

        if is_main_process():
            wandb.log(
                {
                    "lr": lr,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_iou": float(iou),
                    "val_dice": float(dice),
                    "val_acc": float(acc),
                    "val_prec": float(prec),
                    "val_recall": float(recall),
                    "best_iou": float(max_iou),
                    "best_dice": float(max_dice),
                    "best_acc": float(max_acc),
                    "best_prec": float(max_prec),
                    "best_recall": float(max_recall),
                }
            )

    if is_main_process():
        saved_models = glob.glob(save_path + "_*.pt")
        if saved_models:
            latest_model = max(saved_models, key=os.path.getmtime)
            for model_path in saved_models:
                if model_path != latest_model:
                    os.remove(model_path)
        if os.path.exists(os.path.join(output_dir, "checkpoints")):
            shutil.rmtree(os.path.join(output_dir, "checkpoints"))

        wandb.finish()

    logger.info("Training completed")
    cleanup_distributed()


def train_step(config, data, target, model, optimizer, criterion, scaler):
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", enabled=config.train.amp):
        if config.aug.mixup:
            lam = np.random.beta(config.aug.lambdaa, config.aug.lambdaa)
            index = list(torch.randperm(data.size(0)).numpy())
            mixed_x = lam * data + (1 - lam) * data[index]
            output = model(mixed_x)
            loss = lam * criterion(output, target)
            if config.model.ds:
                for x in range(len(target)):
                    target[x] = target[x][index]
                loss = loss + (1 - lam) * criterion(output, target)
            else:
                loss = loss + (1 - lam) * criterion(output, target[index])
        else:
            output = model(data)
            loss = criterion(output, target)

    if config.train.amp and scaler is not None:
        scaler.scale(loss).backward()
        if config.train.clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if config.train.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return loss.detach()


def train(
    config,
    data_loader,
    model,
    optimizer,
    lr_scheduler,
    criterion,
    epoch,
    logger,
    scaler,
):
    model.train()
    loss_meter = AverageMeter()

    if hasattr(data_loader, "sampler") and isinstance(
        data_loader.sampler, DistributedSampler
    ):
        data_loader.sampler.set_epoch(epoch)

    for idx, batch_data in enumerate(data_loader):
        data, target = batch_data["image"], batch_data["label"]
        target = (target > 0.5).float()
        if config.model.ds:
            target = deep_supervision_scale3d(target)
            target = to_cuda(target, gpu_id=torch.cuda.current_device())
        else:
            target = target.cuda()
        data = data.cuda()

        loss = train_step(config, data, target, model, optimizer, criterion, scaler)
        lr_scheduler.step()

        loss_meter.update(loss)
        if (
            idx == 0
            or (idx + 1) % config.train.print_freq == 0
            or idx == len(data_loader) - 1
        ):
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Train:[{epoch+1}/{config.train.epochs}][{idx+1}/{len(data_loader)}]  "
                f"loss:{loss_meter.val:.3f}({loss_meter.avg:.3f})  "
                f"lr:{lr:.6f}"
            )

    loss = reduce_tensor(loss_meter.avg.data)
    return loss, lr


def val_step(config, data, target, model, criterion):
    with autocast(device_type="cuda", enabled=config.train.amp):
        output = model(data)
        if config.model.ds:
            output = output[0]
            target = target[0]
        output = (torch.sigmoid(output) > 0.5).float()
        loss = criterion(output, target)

    metrics = {
        "iou": compute_iou(output, target),
        "dice": compute_dice(output, target),
        "acc": compute_acc(output, target),
        "prec": compute_prec(output, target),
        "recall": compute_recall(output, target),
    }
    return loss, metrics


@torch.inference_mode()
def val(config, data_loader, model, criterion, logger):
    model.eval()
    loss_meter = AverageMeter()
    meters = {
        "iou": AverageMeter(),
        "dice": AverageMeter(),
        "acc": AverageMeter(),
        "prec": AverageMeter(),
        "recall": AverageMeter(),
    }

    for idx, batch_data in enumerate(data_loader):
        data, target = batch_data["image"], batch_data["label"]
        if config.model.ds:
            newtarget = deep_supervision_scale3d(target)
            target = to_cuda(newtarget, gpu_id=torch.cuda.current_device())
        else:
            target = target.cuda()
        data = data.cuda()

        loss, metrics = val_step(config, data, target, model, criterion)

        loss_meter.update(loss)
        for name, value in metrics.items():
            batch_size = (
                target[0].size(0) if isinstance(target, list) else target.size(0)
            )
            meters[name].update(value, batch_size)

        if (
            idx == 0
            or (idx + 1) % config.train.print_freq == 0
            or idx == len(data_loader) - 1
        ):
            log_msg = f"Val:[{idx+1}/{len(data_loader)}]  "
            log_msg += f"loss:{loss_meter.val:.3f}({loss_meter.avg:.3f})  "
            log_msg += "  ".join(
                [f"{k}:{meters[k].val:.3f}({meters[k].avg:.3f})" for k in meters]
            )
            logger.info(log_msg)

    loss = reduce_tensor(loss_meter.avg.data)
    metrics = {k: reduce_tensor(v.avg.data) for k, v in meters.items()}

    return loss, tuple(metrics.values())


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    config = get_config(config_path)
    seed_everything(config.train.seed)

    if is_main_process():
        tojson(config.data.path)

    os.environ["WANDB_API_KEY"] = "38268f0621496e89d1837a26337b64ccc09cd7c7"
    os.environ.setdefault("WANDB_MODE", "disabled" if not config.wandb else "offline")

    if is_main_process():
        wandb.login()

    main(config)
