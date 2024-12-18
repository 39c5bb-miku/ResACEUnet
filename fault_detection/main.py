import os
import random
import datetime
import gc
import warnings
import wandb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from models.build import build_model
from data.build_data import build_loader
from metrics.metrics import AverageMeter,compute_iou,compute_dice,compute_acc,compute_prec,compute_recall,compute_f1
from optimizer.lr_scheduler import build_scheduler
from optimizer.optimizer import build_optimizer
from utils.logger import create_logger
from utils.checkpoint import load_checkpoint, save_checkpoint, auto_resume_helper
from utils.to_torch import to_cuda
from utils.config import get_config
from loss.deep_supervision import MultipleOutputLoss2,deep_supervision_scale3d
from loss.losses import build_loss_fn
from datasets.process import tojson

warnings.filterwarnings("ignore", category=UserWarning)

gc.collect()
torch.cuda.empty_cache()

def build(config):

    data_loader_train,data_loader_val = build_loader(config)
    model = build_model(config)
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    criterion = build_loss_fn(config).cuda()
    if config.model.ds:
        net_numpool = 3
        weights = torch.tensor([1 / (2 ** i) for i in range(net_numpool)])
        weights = weights / weights.sum()
        criterion = MultipleOutputLoss2(criterion, weights).cuda()
    return data_loader_train,data_loader_val,model,optimizer,lr_scheduler,criterion

def main(config):

    output_dir = os.path.join(config.train.output,config.model.name)
    logger = create_logger(output_dir=output_dir,dist_rank=0,name=f"{config.model.name}")
    data_loader_train,data_loader_val,model,optimizer,lr_scheduler,criterion = build(config)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Creating model:{config.model.name}, number of params (M):{(n_parameters / 1.e6):.2f}")
    model.cuda()
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    wandb.init(project=config.model.name,config={
        "lr":config.optimizer.lr,
        "batch_size":config.data.batch_size*config.gpu_num,
        "epochs":config.train.epochs,
        "img_size":config.data.img_size,
        "loss":config.loss.name,
        "optimizer":config.optimizer.name,
        "scheduler":config.lr_scheduler.name,
        "seed":config.train.seed,
    })

    if config.train.auto_resume:
        resume_file = auto_resume_helper(output_dir,logger)
    if resume_file:
        if config.model.resume:
            logger.warning(f"Auto-resume changing resume file from {config.model.resume} to {resume_file}")
        # config.defrost()
        config.model.resume = resume_file
        # config.freeze()
        logger.info(f'Auto resuming from {resume_file}')
    else:
        logger.info(f'No checkpoint found in {output_dir}, ignoring auto resume')
    if config.model.resume:
        load_checkpoint(config, model, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()

    max_iou,max_dice,max_acc = float('-inf'),float('-inf'),float('-inf')
    max_prec,max_recall,max_f1 = float('-inf'),float('-inf'),float('-inf')
    patience,max_patience = 0,config.train.patience
    for epoch in range(config.train.start_epoch, config.train.epochs):
        loss,lr = train(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, logger)
        iou,dice,acc,prec,recall,f1 = val(config, data_loader_val, model, logger)
        if ((epoch+1) % config.train.save_freq == 0):
            save_checkpoint(config, epoch, model, optimizer, lr_scheduler, logger)
        if iou >= max_iou:
            os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
            checkpoints_path = os.path.join(output_dir, "checkpoints", config.model.name)
            logger.info(f'\n'
                f' * iou: {iou:.4f}\n'
                f' * dice: {dice:.4f}\n'
                f' * acc: {acc:.4f}\n'
                f' * prec: {prec:.4f}\n'
                f' * recall: {recall:.4f}\n'
                f' * f1: {f1:.4f}'
            )            
            logger.info(f'Saving best loss model at epoch {epoch+1}')
            torch.save(model.state_dict(), checkpoints_path + '.pt')
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                logger.info("Model hasn't improved for {} epochs. Stopping training".format(max_patience))
                break
        max_iou = max(iou,max_iou)
        max_dice = max(dice, max_dice)
        max_acc = max(acc,max_acc)
        max_prec = max(prec,max_prec)
        max_recall = max(recall,max_recall)
        max_f1 = max(f1,max_f1)
        wandb.log({"lr":lr,"loss":float(loss),
                "val_iou":float(iou),"val_dice":float(dice),"val_acc":float(acc),
                "val_prec":float(prec),"val_recall":float(recall),"val_F1":float(f1),
                "best_iou":float(max_iou),"best_dice":float(max_dice),"best_acc":float(max_acc),
                "best_prec":float(max_prec),"best_recall":float(max_recall),"best_f1":float(max_f1)}
        )

    ender.record()
    torch.cuda.synchronize()
    time = starter.elapsed_time(ender) / 1000
    time = str(datetime.timedelta(seconds=int(time)))
    logger.info('Training time {}'.format(time))   

    wandb.finish()

def train(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, logger):

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    model.train()
    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    scaler = GradScaler()

    for idx,batch_data in enumerate(data_loader):
        samples, targets = batch_data["image"], batch_data["label"]
        targets = (targets > 0.5).float()
        if config.model.ds:
            newtargets = deep_supervision_scale3d(targets)
            targets = to_cuda(newtargets,gpu_id=torch.cuda.current_device())
        else:
            targets = targets.cuda()
        samples = samples.cuda()
        with autocast(device_type='cuda',enabled=config.train.amp):
            if config.aug.mixup:
                lam = np.random.beta(config.aug.lambdaa, config.aug.lambdaa)
                index = list(torch.randperm(samples.size(0)).numpy())
                mixed_x = lam * samples + (1 - lam) * samples[index]
                outputs = model(mixed_x)
                # outputs = torch.sigmoid(outputs)
                loss = lam * criterion(outputs, targets)
                if config.model.ds:
                    for x in range(len(targets)):
                        targets[x]=targets[x][index]
                    loss = loss + (1 - lam) * criterion(outputs, targets)
                else:
                    loss = loss + (1 - lam) * criterion(outputs, targets[index])
            else:
                outputs = model(samples)
                # outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, targets)
        if config.train.amp == True:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.train.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        optimizer.zero_grad()

        lr_scheduler.step_update(epoch * num_steps + idx)
        loss_meter.update(loss)
        if idx == 0 or (idx+1) % config.train.print_freq == 0 or idx == num_steps - 1:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train:[{epoch+1}/{config.train.epochs}][{idx+1}/{num_steps}]  '
                f'lr:{lr:.7f}  '
                f'loss:{loss_meter.val:.4f}({loss_meter.avg:.4f})  '
            )

    loss = loss_meter.avg.data
    ender.record()
    torch.cuda.synchronize()
    time = starter.elapsed_time(ender) / 1000
    logger.info(f"EPOCH {epoch+1} training takes {datetime.timedelta(seconds=int(time))}")

    return loss, lr

@torch.no_grad()
def val(config, data_loader, model, logger):

    model.eval()
    num_steps = len(data_loader)
    iou_meter = AverageMeter()
    dice_meter = AverageMeter()
    acc_meter = AverageMeter()
    prec_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for idx,batch_data in enumerate(data_loader):
        samples, targets = batch_data["image"], batch_data["label"]
        targets[targets>0.5] = 1
        if config.model.ds:
            newtargets = deep_supervision_scale3d(targets)
            targets = to_cuda(newtargets,gpu_id=torch.cuda.current_device())
        else:
            targets = targets.cuda()
        samples = samples.cuda()
        with autocast(device_type='cuda',enabled=config.train.amp):
            starter.record()
            outputs = model(samples)
            ender.record()
            torch.cuda.synchronize()
            time = starter.elapsed_time(ender) / 1000
            if config.model.ds:
                outputs = outputs[0]
                targets = targets[0]
            outputs = (torch.sigmoid(outputs) > 0.5).float()

        iou = compute_iou(outputs, targets)
        dice = compute_dice(outputs, targets)
        acc = compute_acc(outputs, targets)
        prec = compute_prec(outputs, targets)
        recall = compute_recall(outputs, targets)
        f1 = compute_f1(outputs, targets)

        iou_meter.update(iou, targets.size(0))
        dice_meter.update(dice, targets.size(0))
        acc_meter.update(acc, targets.size(0))
        prec_meter.update(prec, targets.size(0))
        recall_meter.update(recall, targets.size(0))
        f1_meter.update(f1, targets.size(0))

        if idx == 0 or (idx+1) % config.train.print_freq == 0 or idx == num_steps - 1:
            logger.info(
                f'Val:[{idx+1}/{num_steps}]  '
                f'Inference Time:{time:.3f}s  '
                f'iou:{iou_meter.val:.4f}({iou_meter.avg:.4f})  '
                f'dice:{dice_meter.val:.4f}({dice_meter.avg:.4f})  '
                f'acc:{acc_meter.val:.4f}({acc_meter.avg:.4f})  '
                f'prec:{prec_meter.val:.4f}({prec_meter.avg:.4f})  '
                f'recall:{recall_meter.val:.4f}({recall_meter.avg:.4f})  '
                f'f1:{f1_meter.val:.4f}({f1_meter.avg:.4f})  '
            )

    iou = iou_meter.avg.data
    dice = dice_meter.avg.data
    acc = acc_meter.avg.data
    prec = prec_meter.avg.data
    recall = recall_meter.avg.data
    f1 = f1_meter.avg.data

    return iou, dice, acc, prec, recall, f1

def seed_everything(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

if __name__ == '__main__':

    config_path = 'configs/config.yaml'
    config = get_config(config_path)
    seed_everything(config.train.seed)
    tojson(config.data.path)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    os.environ["WANDB_API_KEY"] = ''
    os.environ.setdefault("WANDB_MODE", "disabled" if not config.wandb else "offline")
    wandb.login()
    main(config)

