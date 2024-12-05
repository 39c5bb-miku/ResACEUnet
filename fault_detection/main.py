import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import time
import json
import datetime
import gc
import warnings
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from models.build import build_model
from data.build_data import build_loader
from metrics.metrics import AverageMeter,compute_iou,compute_dice,compute_acc,compute_prec,compute_recall,compute_f1
from optimizer.lr_scheduler import build_scheduler
from optimizer.optimizer import build_optimizer
from utils.logger import create_logger
from utils.checkpoint import load_checkpoint, save_checkpoint, auto_resume_helper
from utils.to_torch import to_cuda
from utils.conf import get_conf
from loss.deep_supervision import MultipleOutputLoss2,deep_supervision_scale3d
from monai.losses import DiceCELoss
from datasets.process import tojson

warnings.filterwarnings("ignore", category=UserWarning)

gc.collect()
torch.cuda.empty_cache()

def init_ddp(local_rank):

    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

def reduce_tensor(tensor: torch.Tensor):

    rt = tensor.clone()  
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def main(local_rank,config):

    init_ddp(local_rank)
    output_dir = os.path.join(config.train.output,config.model.name)
    path = os.path.join(output_dir, "config.json")
    logger = create_logger(output_dir=output_dir,dist_rank=local_rank,name=f"{config.model.name}")
    logger.info(f"Full config saved to {path}")
    data_loader_train,data_loader_val = build_loader(config)

    logger.info(f"Creating model:{config.model.name}")
    model = build_model(config)
    model.cuda()
    if config.gpu_num > 1:
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model,device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    if config.loss.name == 'ce_dice_loss':
        criterion = DiceCELoss(to_onehot_y=True, lambda_ce=config.loss.ce_weight, lambda_dice=config.loss.dice_weight).cuda()
    if config.model.ds:
        net_numpool = 3
        weights = torch.tensor([1 / (2 ** i) for i in range(net_numpool)])
        weights = weights / weights.sum()
        criterion = MultipleOutputLoss2(criterion, weights).cuda()

    if local_rank == 0:
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
            logger.warning(f"auto-resume changing resume file from {config.model.resume} to {resume_file}")
        # config.defrost()
        config.model.resume = resume_file
        # config.freeze()
        logger.info(f'auto resuming from {resume_file}')
    else:
        logger.info(f'no checkpoint found in {output_dir}, ignoring auto resume')
    if config.model.resume:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()

    max_iou = float('-inf')
    max_dice = float('-inf')
    max_acc = float('-inf')
    max_prec = float('-inf')
    max_recall = float('-inf')
    max_f1 = float('-inf')
    min_loss = float('inf')
    patience = 0  
    max_patience = config.train.patience  
    for epoch in range(config.train.start_epoch, config.train.epochs):
        loss,lr = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, logger)
        iou,dice,acc,prec,recall,f1,loss1 = val(config, data_loader_val, model, criterion, logger)
        if ((epoch+1) % config.train.save_freq == 0) and local_rank == 0:
            save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger)
        if local_rank == 0:
            max_iou = max(iou,max_iou)
            max_dice = max(dice, max_dice)
            max_acc = max(acc,max_acc)
            max_prec = max(prec,max_prec)
            max_recall = max(recall,max_recall)
            max_f1 = max(f1,max_f1)
            wandb.log({"lr":lr,"val_iou":iou,"val_dice":dice,"val_acc":acc,"val_prec":prec,"val_recall":recall,"val_F1":f1,
                    "train_loss":loss,"val_loss":loss1,
                    "best_iou":max_iou,"best_dice":max_dice,"best_acc":max_acc,"best_prec":max_prec,"best_recall":max_recall,"best_f1":max_f1})
        if loss1 <= min_loss:
            min_loss = min(min_loss, loss1)
            os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
            checkpoints_path = os.path.join(output_dir, "checkpoints", config.model.name)
            logger.info(f' * iou: {iou:.4f}')
            logger.info(f' * dice: {dice:.4f}')
            logger.info(f' * acc: {acc:.4f}')
            logger.info(f' * prec: {prec:.4f}')
            logger.info(f' * recall: {recall:.4f}')
            logger.info(f' * f1: {f1:.4f}')
            logger.info(f'Saving best loss model, min loss: {min_loss:.4f}')
            if local_rank == 0:
                torch.save(model.state_dict(), checkpoints_path + '.pt')
            patience = 0  
        else:
            patience += 1  
            if patience >= max_patience:
                logger.info("loss hasn't improved for {} epochs. Stopping training.".format(max_patience))
                break
        continue
    dist.destroy_process_group()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('training time {}'.format(total_time_str))   

    wandb.finish()

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, logger):

    start = time.time()
    model.train()
    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    for idx,batch_data in enumerate(data_loader):
        end = time.time()
        samples, targets = batch_data["image"], batch_data["label"]
        targets = (targets > 0.5).float()
        if config.model.ds:
            newtargets = deep_supervision_scale3d(targets)
            targets = to_cuda(newtargets,gpu_id=torch.cuda.current_device())
        else:
            targets = targets.cuda()
        samples = samples.cuda()
        with autocast(enabled=config.train.amp):
            if config.aug.mixup:
                lam = np.random.beta(config.aug.lambdaa, config.aug.lambdaa)
                index = torch.randperm(samples.size(0)).cuda()
                mixed_x = lam * samples + (1 - lam) * samples[index]
                outputs = model(mixed_x)
                loss = lam * criterion(outputs, targets)
                if config.model.ds:
                    for x in range(len(targets)):
                        targets[x]=targets[x][index]
                    loss = loss + (1 - lam) * criterion(outputs, targets)
                    outputs = outputs[0]
                    targets = targets[0]
                else:
                    loss = loss + (1 - lam) * criterion(outputs, targets[index])
            else:
                outputs = model(samples)
                loss = criterion(outputs, targets)
                if config.model.ds:
                    outputs = outputs[0]
                    targets = targets[0]
        if config.train.amp == True:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.train.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
            optimizer.step()
        optimizer.zero_grad()

        lr_scheduler.step_update(epoch * num_steps + idx)
        loss_meter.update(loss)
        batch_time.update(time.time() - end)

        if idx == 0 or (idx+1) % config.train.print_freq == 0 or idx == num_steps - 1:
            lr = optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'train:[{epoch+1}/{config.train.epochs}][{idx+1}/{num_steps}]  '
                f'eta:{datetime.timedelta(seconds=int(etas))}  '
                f'lr:{lr:.7f}  '
                f'time:{batch_time.val:.4f}({batch_time.avg:.4f})  '
                f'loss:{loss_meter.val:.4f}({loss_meter.avg:.4f})  ')

    loss = reduce_tensor(loss_meter.avg.data)
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch+1} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return loss, lr

@torch.no_grad()
def val(config, data_loader, model, criterion, logger):

    model.eval()
    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    iou_meter = AverageMeter()
    dice_meter = AverageMeter()
    acc_meter = AverageMeter()
    prec_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()

    for idx,batch_data in enumerate(data_loader):
        end = time.time()
        samples, targets = batch_data["image"], batch_data["label"]
        targets[targets>0.5] = 1
        if config.model.ds:
            newtargets = deep_supervision_scale3d(targets)
            targets = to_cuda(newtargets,gpu_id=torch.cuda.current_device())
        else:
            targets = targets.cuda()
        samples = samples.cuda()
        with torch.no_grad():
            with autocast(enabled=config.train.amp):
                outputs = model(samples)
                loss = criterion(outputs, targets)
                if config.model.ds:
                    outputs = outputs[0]
                    targets = targets[0]

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

        loss_meter.update(loss)
        batch_time.update(time.time() - end)

        if idx == 0 or (idx+1) % config.train.print_freq == 0 or idx == num_steps - 1:
            logger.info(
                f'Val:[{idx+1}/{num_steps}]  '
                f'time:{batch_time.val:.4f}({batch_time.avg:.4f})  '
                f'iou:{iou_meter.val:.4f}({iou_meter.avg:.4f})  '
                f'dice:{dice_meter.val:.4f}({dice_meter.avg:.4f})  '
                f'acc:{acc_meter.val:.4f}({acc_meter.avg:.4f})  '
                f'prec:{prec_meter.val:.4f}({prec_meter.avg:.4f})  '
                f'recall:{recall_meter.val:.4f}({recall_meter.avg:.4f})  '
                f'f1:{f1_meter.val:.4f}({f1_meter.avg:.4f})  '
                f'loss:{loss_meter.val:.4f}({loss_meter.avg:.4f})  ')

    iou = reduce_tensor(iou_meter.avg.data)
    dice = reduce_tensor(dice_meter.avg.data)
    acc = reduce_tensor(acc_meter.avg.data)
    prec = reduce_tensor(prec_meter.avg.data)
    recall = reduce_tensor(recall_meter.avg.data)
    f1 = reduce_tensor(f1_meter.avg.data)
    loss = reduce_tensor(loss_meter.avg.data)

    return iou, dice, acc, prec, recall, f1, loss

if __name__ == '__main__':
    config_path = 'configs/config.yaml'
    config, config_dict = get_conf(config_path)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '19199'
    world_size = config.gpu_num
    os.environ['WORLD_SIZE'] = str(world_size)
    tojson(config.data.path)
    os.environ["WANDB_API_KEY"] = ''
    os.environ.setdefault("WANDB_MODE", "disabled" if not config.wandb else "offline")
    wandb.login()
    seed = config.train.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True
    cudnn.enabled = True
    output_dir = os.path.join(config.train.output,config.model.name)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.json")
    with open(path, "w") as f:
        f.write(json.dumps(config_dict, indent=4))
    mp.spawn(fn=main, args=(config, ), nprocs=world_size)
