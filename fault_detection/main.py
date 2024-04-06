import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"#设置GPU的编号
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#设置使用的GPU编号
import time
import argparse
import datetime
import gc
import warnings
import numpy as np
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from metrics import AverageMeter,compute_iou,compute_dice,compute_acc,compute_prec,compute_recall,compute_f1,bce_dice_loss
from config import get_config
from models.build import build_model
from data.build_data import get_loader as build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils.logger import create_logger
from utils.to_torch import to_cuda
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.utils import get_grad_norm, auto_resume_helper
from loss.dice_loss import DC_and_BCE_loss, DC_and_CE_loss, DC_and_topk_loss
from loss.deep_supervision import MultipleOutputLoss2, deep_supervision_scale3d

warnings.filterwarnings("ignore", category=UserWarning)

gc.collect()
torch.cuda.empty_cache()

scaler = GradScaler()

def parse_option():

    parser = argparse.ArgumentParser('Transformer training and evaluation script', add_help=False)#
    parser.add_argument('--cfg', type=str, metavar="FILE", default='configs/UNETR_PP4.yaml',help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    
    return args, config


_, config = parse_option()


def main(config=config,fine_tune=None,using_pretrain=None):

    dataset_train, dataset_val, data_loader_train, data_loader_val= build_loader(config)#训练数据集，验证数据集，训练数据加载器，验证数据加载器
    logger.info(f"Creating model:{config.MODEL.NAME}")#创建模型
    model = build_model(config)
    model.cuda()#在 GPU 上进行训练和推理

    optimizer = build_optimizer(config, model)#创建了一个优化器，以便用于模型的训练
    model = torch.nn.DataParallel(model).cuda()#将模型自动划分到所有可用的 GPU 上，并将数据划分为多个批次分别在不同的 GPU 上进行计算
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))#构建学习率调度器

    if config.TRAIN.LOSS.NAME == 'bce_dice_loss':
        # criterion = DC_and_BCE_loss({}, {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False})
        criterion = bce_dice_loss(bce_weight=0.5, smooth_nr=0., smooth_dr=1e-6,smoothing=0.05)
    elif config.TRAIN.LOSS.NAME == 'ce_dice_loss':
        criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
    elif config.TRAIN.LOSS.NAME == 'topk_dice_loss':
        criterion = DC_and_topk_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

    if config.MODEL.DS:
        net_numpool = 3
        weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
        weights = weights / weights.sum()
        criterion = MultipleOutputLoss2(criterion, weights)

    # wandb.watch(model, criterion, log="all", log_freq=2)#监控模型
    max_iou = 0.0#当前最大的交并比
    max_dice = 0.0
    max_acc = 0.0#当前最大的准确率
    max_prec = 0.0
    max_recall = 0.0
    max_f1 = 0.0#当前最大的F1分数
    min_loss=2.0#追踪模型训练过程中的最小损失值
    patience = 0  # 初始化耐心计数器
    max_patience = config.TRAIN.PATIENCE  # 定义最大耐心值

    if config.TRAIN.AUTO_RESUME:#检查是否启用了自动恢复功能
        resume_file = auto_resume_helper(config.OUTPUT,logger)#自动恢复功能的帮助函数
        if resume_file:#如果有之前训练的模型需要恢复
            if config.MODEL.RESUME:#
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()#解冻配置
            config.MODEL.RESUME = resume_file#恢复模型
            config.freeze()#冻结配置
            logger.info(f'auto resuming from {resume_file}')
        else:#如果没有之前训练的模型需要恢复
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:#检查是否有之前训练的模型需要恢复
        max_iou = load_checkpoint(config, model, optimizer, lr_scheduler, logger)#加载之前训练的模型
        iou,dice,acc,prec,recall,f1,loss1  = validate(config, data_loader_val, model,criterion)#使用验证集对模型进行验证
        logger.info(f"Iou of the network on the {len(dataset_val)} test images: {iou:.1f}% {acc:.1f}% {f1:.1f}%")
        logger.info(f'Max iou: {max_iou:.3f}')#打印当前模型的最佳交并比
        if config.EVAL_MODE:#如果是评估模式
            return

    logger.info("Start training")#开始训练
    start_time = time.time()#记录训练开始的时间

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):#从起始轮次 config.TRAIN.START_EPOCH 开始,到训练轮次 config.TRAIN.EPOCHS 结束
        loss,lr = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler)
        if  (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model, max_iou, optimizer, lr_scheduler, logger)
        iou,dice,acc,prec,recall,f1,loss1 = validate(config, data_loader_val, model,criterion)
        max_iou=max(iou,max_iou)#更新最佳交并比
        max_dice = max(dice, max_dice)#更新最佳dice系数
        max_acc=max(acc,max_acc)#更新最佳准确率
        max_prec=max(prec,max_prec)#更新最佳精确率
        max_recall=max(recall,max_recall)#更新最佳召回率
        max_f1=max(f1,max_f1)#更新最佳F1分数
        logger.info(f"Iou of the network on the {len(dataset_val)} test images: {iou:.1f}% {acc:.1f}% {f1:.1f}%")
        logger.info(f'Max iou: {max_iou:.3f}')#打印当前模型的最佳交并比
        if loss1 <= min_loss:#如果当前的损失小于最小损失，那么就将当前模型的权重保存下来
            min_loss = min(min_loss, loss1)
            logger.info(f'Saving best model, max_iou: {max_iou:.3f}')
            torch.save(model.state_dict(), config.MODEL.NAME + '.pt')#保存当前模型的权重
            patience = 0  # 重置耐心计数器
        else:
            patience += 1  # 更新耐心计数器
            if patience >= max_patience:
                logger.info("Validation loss hasn't improved for {} epochs. Stopping training.".format(max_patience))
                break
        continue

    total_time = time.time() - start_time#计算训练的总时间
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))#将训练的总时间转换为字符串
    logger.info('Training time {}'.format(total_time_str))   


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler):#训练函数

    model.train()#将模型设置为训练模式
    num_steps = len(data_loader)#获取数据集的长度
    batch_time = AverageMeter()#记录每个批次的时间
    loss_meter = AverageMeter()#记录每个批次的损失
    norm_meter = AverageMeter()#记录每个批次的归一化损失
    iou_meter = AverageMeter()#记录每个批次的交并比
    dice_meter = AverageMeter()#记录每个批次的dice系数
    acc_meter = AverageMeter()#记录每个批次的准确率
    prec_meter = AverageMeter()#记录每个批次的精确率
    recall_meter = AverageMeter()#记录每个批次的召回率
    f1_meter = AverageMeter()#记录每个批次的F1分数
    start = time.time()#记录训练开始的时间
    end = time.time()#记录训练结束的时间

    for idx,batch_data in enumerate(data_loader):#获取批次编号idx和批次数据batch_data
        if isinstance(batch_data, list):#如果batch_data是列表类型
            samples, targets = batch_data#获取输入数据samples和标签数据targets
        else:#如果batch_data是字典类型
            samples, targets = batch_data["image"], batch_data["label"]#获取输入数据samples和标签数据targets
        targets = (targets > 0.7).float()
        if config.MODEL.DS:
            newtargets = deep_supervision_scale3d(targets)
            targets = to_cuda(newtargets)
        else:
            targets = targets.cuda()
        samples = samples.cuda()
        with autocast():#使用自动混合精度训练
            if config.AUG.MIXUP:
                lam = np.random.beta(config.TRAIN.LAMBDA, config.TRAIN.LAMBDA)#生成一个beta分布随机数作为数据增强的参数lam
                index = torch.randperm(samples.size(0)).cuda()#生成一个长度为批次大小的随机序列，并移动到GPU上
                mixed_x = lam * samples + (1 - lam) * samples[index]#对输入数据进行mixup数据增强
                outputs = model(mixed_x)#将增强后的输入数据输入到模型中
                loss = lam * criterion(outputs, targets)
                if config.MODEL.DS:
                    for x in range(len(targets)):
                        targets[x]=targets[x][index]
                    loss = loss + (1 - lam) * criterion(outputs, targets)
                    outputs = outputs[0]
                    targets = targets[0]
                else:
                    loss = loss + (1 - lam) * criterion(outputs, targets[index])
                iou = lam * compute_iou(outputs, targets) + (1 - lam) * compute_iou(outputs, targets[index])
                dice = lam * compute_dice(outputs, targets) + (1 - lam) * compute_dice(outputs, targets[index])
                acc = lam * compute_acc(outputs, targets) + (1 - lam) * compute_acc(outputs, targets[index])
                prec = lam * compute_prec(outputs, targets) + (1 - lam) * compute_prec(outputs, targets[index])
                recall = lam * compute_recall(outputs, targets) + (1 - lam) * compute_recall(outputs, targets[index])
                f1 = lam * compute_f1(outputs, targets) + (1 - lam) * compute_f1(outputs, targets[index])
            else:#不进行数据增强
                outputs = model(samples)#将输入数据输入到模型中
                loss = criterion(outputs, targets)#计算模型输出的损失值loss
                if config.MODEL.DS:
                    outputs = outputs[0]
                    targets = targets[0]
                iou = compute_iou(outputs, targets)#计算模型输出的交并比值iou
                dice = compute_dice(outputs, targets)#计算模型输出的dice系数
                acc = compute_acc(outputs, targets)#计算模型输出的准确率值acc
                prec = compute_prec(outputs, targets)#计算模型输出的精确率值prec
                recall = compute_recall(outputs, targets)#计算模型输出的召回率值recall
                f1 = compute_f1(outputs, targets)#计算模型输出的F1分数
        
        if config.TRAIN.AMP:
            scaler.scale(loss).backward()#反向传播
        else:
            loss.backward()#反向传播

        if config.TRAIN.CLIP_GRAD:#如果设置了梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)#计算梯度范数
        else:#如果没有设置梯度裁剪
            grad_norm = get_grad_norm(model.parameters())#计算梯度范数

        if config.TRAIN.AMP:
            scaler.step(optimizer)#更新模型参数
            scaler.update()#更新scaler
            optimizer.zero_grad()#清空梯度
        else:
            optimizer.step()#更新模型参数
            optimizer.zero_grad()#清空梯度
        
        lr_scheduler.step_update(epoch * num_steps + idx)#更新学习率
        iou_meter.update(iou, targets.size(0))#更新交并比
        dice_meter.update(dice, targets.size(0))#更新dice系数
        acc_meter.update(acc, targets.size(0))#更新准确率
        prec_meter.update(prec, targets.size(0))#更新精确率
        recall_meter.update(recall, targets.size(0))#更新召回率
        f1_meter.update(f1, targets.size(0))#更新F1分数
        loss_meter.update(loss.item(), targets.size(0))#更新损失
        norm_meter.update(grad_norm)#更新梯度范数
        batch_time.update(time.time() - end)#更新每个batch的时间
        end = time.time()#记录训练结束的时间

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']#获取当前的学习率
            etas = batch_time.avg * (num_steps - idx)#计算当前训练的剩余时间
            logger.info(
                f'Train:[{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  '
                f'eta:{datetime.timedelta(seconds=int(etas))}  '
                f'lr:{lr:.8f}  '
                f'time:{batch_time.val:.3f}({batch_time.avg:.3f})  '
                f'loss:{loss_meter.val:.4f}({loss_meter.avg:.2f})  '
                f'iou:{iou_meter.val:.4f}({iou_meter.avg:.2f})  '
                f'dice:{dice_meter.val:.4f}({dice_meter.avg:.2f})  '
                f'acc:{acc_meter.val:.4f}({acc_meter.avg:.2f})  '
                f'prec:{prec_meter.val:.4f}({prec_meter.avg:.2f})  '
                f'recall:{recall_meter.val:.4f}({recall_meter.avg:.2f})  '
                f'f1:{f1_meter.val:.4f}({f1_meter.avg:.2f})  '
                f'grad_norm {norm_meter.val:.4f}({norm_meter.avg:.2f})  ')

    epoch_time = time.time() - start#计算当前epoch的训练时间
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return loss_meter.avg, lr#返回平均损失和当前学习率


@torch.no_grad()#不计算梯度
def validate(config, data_loader, model,criterion):#验证函数

    model.eval()#将模型设置为验证模式
    batch_time = AverageMeter()#初始化每个批次的时间
    loss_meter = AverageMeter()#初始化损失
    iou_meter = AverageMeter()#初始化交并比
    dice_meter = AverageMeter()#初始化dice系数
    acc_meter = AverageMeter()#初始化准确率
    prec_meter = AverageMeter()#初始化精确率
    recall_meter = AverageMeter()#初始化召回率
    f1_meter = AverageMeter()#初始化F1分数
    end = time.time()#记录验证开始的时间

    for idx,batch_data in enumerate(data_loader):#获取批次编号idx和批次数据batch_data
        if isinstance(batch_data, list):#如果batch_data是列表类型
            samples, targets = batch_data#获取输入数据samples和标签数据targets
        else:#如果batch_data是字典类型
            samples, targets = batch_data["image"], batch_data["label"]#获取输入数据samples和标签数据targets
        targets[targets>0.7] = 1
        if config.MODEL.DS:
            newtargets = deep_supervision_scale3d(targets)
            targets = to_cuda(newtargets)
        else:
            targets = targets.cuda()        
        samples = samples.cuda()
        outputs = model(samples)
        loss = criterion(outputs, targets)#计算损失
        if config.MODEL.DS:
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
        loss_meter.update(loss.item(), targets.size(0))#更新损失
        batch_time.update(time.time() - end)#更新每个batch的时间
        end = time.time()#记录验证结束的时间
        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f'Val:[{idx}/{len(data_loader)}]  '
                f'time:{batch_time.val:.3f}({batch_time.avg:.3f})  '
                f'loss:{loss_meter.val:.4f}({loss_meter.avg:.2f})  '
                f'iou:{iou_meter.val:.4f}({iou_meter.avg:.2f})  '
                f'dice:{dice_meter.val:.4f}({dice_meter.avg:.2f})  '
                f'acc:{acc_meter.val:.4f}({acc_meter.avg:.2f})  '
                f'prec:{prec_meter.val:.4f}({prec_meter.avg:.2f})  '
                f'recall:{recall_meter.val:.4f}({recall_meter.avg:.2f})  '
                f'f1:{f1_meter.val:.4f}({f1_meter.avg:.2f})  ')

    logger.info(f' * iou: {iou_meter.avg:.3f} ')
    logger.info(f' * dice: {dice_meter.avg:.3f} ')
    logger.info(f' * acc: {acc_meter.avg:.3f} ')
    logger.info(f' * prec: {prec_meter.avg:.3f} ')
    logger.info(f' * recall: {recall_meter.avg:.3f} ')
    logger.info(f' * f1: {f1_meter.avg:.3f} ')

    return iou_meter.avg, dice_meter.avg, acc_meter.avg, prec_meter.avg, recall_meter.avg, f1_meter.avg, loss_meter.avg


if __name__ == '__main__':

    seed = config.SEED#设置随机种子
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True
    cudnn.enabled = True
    os.makedirs(config.OUTPUT, exist_ok=True)#创建输出文件夹
    logger = create_logger(output_dir=config.OUTPUT,name=f"{config.MODEL.NAME}")
    path = os.path.join(config.OUTPUT, "config.json")#设置配置文件的保存路径

    with open(path, "w") as f:#保存配置文件
        f.write(config.dump())#保存配置文件

    logger.info(f"Full config saved to {path}")
    # logger.info(config.dump())
    main()
