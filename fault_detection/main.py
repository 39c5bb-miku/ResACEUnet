import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"#设置GPU的编号
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'#设置使用的GPU编号
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
from collections import OrderedDict
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
    path = os.path.join(output_dir, "config.json")#设置配置文件的保存路径
    logger = create_logger(output_dir=output_dir,dist_rank=local_rank,name=f"{config.model.name}")
    logger.info(f"Full config saved to {path}")
    data_loader_train,data_loader_val = build_loader(config)

    logger.info(f"Creating model:{config.model.name}")#创建模型
    model = build_model(config)
    # checkpoint = torch.load("ckpt_epoch_301.pth", map_location='cpu')['model']
    # new_ckpt = OrderedDict()
    # for key in list(checkpoint.keys()):
    #     if not (key.startswith("decoder") or key.startswith("mask")):
    #         new_ckpt[key] = checkpoint[key]
    # msg = model.load_state_dict(new_ckpt, strict=False)
    # logger.info(msg)
    model.cuda()
    if config.gpu_num > 1:
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model,device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))

    optimizer = build_optimizer(config, model)#创建了一个优化器，以便用于模型的训练
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))#构建学习率调度器
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

    if config.train.auto_resume:#检查是否启用了自动恢复功能
        resume_file = auto_resume_helper(output_dir,logger)#自动恢复功能的帮助函数
    if resume_file:#如果有之前训练的模型需要恢复
        if config.model.resume:#
            logger.warning(f"auto-resume changing resume file from {config.model.resume} to {resume_file}")
        # config.defrost()#解冻配置
        config.model.resume = resume_file#恢复模型
        # config.freeze()#冻结配置
        logger.info(f'auto resuming from {resume_file}')
    else:#如果没有之前训练的模型需要恢复
        logger.info(f'no checkpoint found in {output_dir}, ignoring auto resume')
    if config.model.resume:#检查是否有之前训练的模型需要恢复
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)#加载之前训练的模型

    logger.info("Start training")#开始训练
    start_time = time.time()#记录训练开始的时间

    max_iou = float('-inf')
    max_dice = float('-inf')
    max_acc = float('-inf')
    max_prec = float('-inf')
    max_recall = float('-inf')
    max_f1 = float('-inf')
    min_loss = float('inf')
    patience = 0  # 初始化耐心计数器
    max_patience = config.train.patience  # 定义最大耐心值
    for epoch in range(config.train.start_epoch, config.train.epochs):
        loss,lr = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, logger)
        iou,dice,acc,prec,recall,f1,loss1 = val(config, data_loader_val, model, criterion, logger)
        if ((epoch+1) % config.train.save_freq == 0) and local_rank == 0:
            save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger)
        if local_rank == 0:
            max_iou = max(iou,max_iou)#更新最佳交并比
            max_dice = max(dice, max_dice)#更新最佳dice系数
            max_acc = max(acc,max_acc)#更新最佳准确率
            max_prec = max(prec,max_prec)#更新最佳精确率
            max_recall = max(recall,max_recall)#更新最佳召回率
            max_f1 = max(f1,max_f1)#更新最佳F1分数
            wandb.log({"lr":lr,"val_iou":iou,"val_dice":dice,"val_acc":acc,"val_prec":prec,"val_recall":recall,"val_F1":f1,
                    "train_loss":loss,"val_loss":loss1,
                    "best_iou":max_iou,"best_dice":max_dice,"best_acc":max_acc,"best_prec":max_prec,"best_recall":max_recall,"best_f1":max_f1})
        if loss1 <= min_loss:#如果当前的损失小于最小损失，那么就将当前模型的权重保存下来
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
                torch.save(model.state_dict(), checkpoints_path + '.pt')#保存当前模型的权重
            patience = 0  # 重置耐心计数器
        else:
            patience += 1  # 更新耐心计数器
            if patience >= max_patience:
                logger.info("loss hasn't improved for {} epochs. Stopping training.".format(max_patience))
                break
        continue
    dist.destroy_process_group()

    total_time = time.time() - start_time#计算训练的总时间
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))#将训练的总时间转换为字符串
    logger.info('training time {}'.format(total_time_str))   

    wandb.finish()#结束wandb

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, logger):

    start = time.time()#记录训练开始的时间
    model.train()
    num_steps = len(data_loader)#获取数据集的长度
    loss_meter = AverageMeter()
    batch_time = AverageMeter()#记录每个批次的时间
    scaler = GradScaler()

    for idx,batch_data in enumerate(data_loader):#获取批次编号idx和批次数据batch_data
        end = time.time()
        samples, targets = batch_data["image"], batch_data["label"]#获取输入数据samples和标签数据targets
        targets = (targets > 0.5).float()
        if config.model.ds:
            newtargets = deep_supervision_scale3d(targets)
            targets = to_cuda(newtargets,gpu_id=torch.cuda.current_device())
        else:
            targets = targets.cuda()
        samples = samples.cuda()
        with autocast(enabled=config.train.amp):#使用自动混合精度训练
            if config.aug.mixup:
                lam = np.random.beta(config.aug.lambdaa, config.aug.lambdaa)
                index = torch.randperm(samples.size(0)).cuda()#生成一个长度为批次大小的随机序列，并移动到GPU上
                mixed_x = lam * samples + (1 - lam) * samples[index]#对输入数据进行mixup数据增强
                outputs = model(mixed_x)#将增强后的输入数据输入到模型中
                loss = lam * criterion(outputs, targets)
                if config.model.ds:
                    for x in range(len(targets)):
                        targets[x]=targets[x][index]
                    loss = loss + (1 - lam) * criterion(outputs, targets)
                    outputs = outputs[0]
                    targets = targets[0]
                else:
                    loss = loss + (1 - lam) * criterion(outputs, targets[index])
            else:#不进行数据增强
                outputs = model(samples)#将输入数据输入到模型中
                loss = criterion(outputs, targets)#计算模型输出的损失值loss
                if config.model.ds:
                    outputs = outputs[0]
                    targets = targets[0]
        if config.train.amp == True:
            scaler.scale(loss).backward()#反向传播
            scaler.step(optimizer)#更新模型参数
            scaler.update()#更新scaler
        else:
            loss.backward()#反向传播
            if config.train.clip_grad:#如果设置了梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)#计算梯度范数
            optimizer.step()#更新模型参数
        optimizer.zero_grad()

        lr_scheduler.step_update(epoch * num_steps + idx)#更新学习率
        loss_meter.update(loss)
        batch_time.update(time.time() - end)#更新每个batch的时间

        if idx == 0 or (idx+1) % config.train.print_freq == 0 or idx == num_steps - 1:
            lr = optimizer.param_groups[0]['lr']#获取当前的学习率
            etas = batch_time.avg * (num_steps - idx)#计算当前训练的剩余时间
            logger.info(
                f'train:[{epoch+1}/{config.train.epochs}][{idx+1}/{num_steps}]  '
                f'eta:{datetime.timedelta(seconds=int(etas))}  '
                f'lr:{lr:.7f}  '
                f'time:{batch_time.val:.4f}({batch_time.avg:.4f})  '
                f'loss:{loss_meter.val:.4f}({loss_meter.avg:.4f})  ')

    loss = reduce_tensor(loss_meter.avg.data)
    epoch_time = time.time() - start#计算当前epoch的训练时间
    logger.info(f"EPOCH {epoch+1} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return loss, lr#返回平均损失和当前学习率

@torch.no_grad()#不计算梯度
def val(config, data_loader, model, criterion, logger):#验证函数

    model.eval()#将模型设置为验证模式
    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    batch_time = AverageMeter()#初始化每个批次的时间
    iou_meter = AverageMeter()#初始化交并比
    dice_meter = AverageMeter()#初始化dice系数
    acc_meter = AverageMeter()#初始化准确率
    prec_meter = AverageMeter()#初始化精确率
    recall_meter = AverageMeter()#初始化召回率
    f1_meter = AverageMeter()#初始化F1分数

    for idx,batch_data in enumerate(data_loader):#获取批次编号idx和批次数据batch_data
        end = time.time()#记录验证结束的时间
        samples, targets = batch_data["image"], batch_data["label"]#获取输入数据samples和标签数据targets
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

        iou = compute_iou(outputs, targets)#计算交并比
        dice = compute_dice(outputs, targets)#计算dice系数
        acc = compute_acc(outputs, targets)#计算准确率
        prec = compute_prec(outputs, targets)#计算精确率
        recall = compute_recall(outputs, targets)#计算召回率
        f1 = compute_f1(outputs, targets)#计算F1分数

        iou_meter.update(iou, targets.size(0))#更新交并比
        dice_meter.update(dice, targets.size(0))#更新dice系数
        acc_meter.update(acc, targets.size(0))#更新准确率
        prec_meter.update(prec, targets.size(0))#更新精确率
        recall_meter.update(recall, targets.size(0))#更新召回率
        f1_meter.update(f1, targets.size(0))#更新F1分数

        loss_meter.update(loss)
        batch_time.update(time.time() - end)#更新每个batch的时间

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
    os.environ['MASTER_ADDR'] = 'localhost'  # 0号机器的IP
    os.environ['MASTER_PORT'] = '19199'  # 0号机器的可用端口
    world_size = config.gpu_num
    os.environ['WORLD_SIZE'] = str(world_size)
    tojson(config.data.path)
    os.environ["WANDB_API_KEY"] = '38268f0621496e89d1837a26337b64ccc09cd7c7'
    os.environ.setdefault("WANDB_MODE", "disabled" if not config.wandb else "offline")
    wandb.login()# 登录wandb
    seed = config.train.seed#设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True
    cudnn.enabled = True
    output_dir = os.path.join(config.train.output,config.model.name)
    os.makedirs(output_dir, exist_ok=True)#创建输出文件夹
    path = os.path.join(output_dir, "config.json")#设置配置文件的保存路径
    with open(path, "w") as f:
        f.write(json.dumps(config_dict, indent=4))
    mp.spawn(fn=main, args=(config, ), nprocs=world_size)
