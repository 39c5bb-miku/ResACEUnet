import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 8
# Input image size
_C.DATA.IMG_SIZE = 96
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = False
# Number of data loading threads
_C.DATA.NUM_WORKERS = 2
# 135-34 200-20
_C.DATA.DATA_PATH='datasets/200-20/'
# 
_C.DATA.JSON_PATH=''
# dataset.json
_C.DATA.JSON_NAME='dataset.json'
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
#'UNET'  'UNETR'  'SWIN_UNETR'  'ACC_UNET'  'UNETR_PP'  'UXNet'   'SEGMAMBA'
_C.MODEL.NAME = 'RESACEUNET' 
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Deep supervision
_C.MODEL.DS = False
# Number of input channels
_C.MODEL.IN_CHANS = 1
# Number of dimensions
_C.MODEL.NUM_POOLS = 1
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1
# Dropout rate
_C.MODEL.DROP_RATE = 0.1
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# amp
_C.TRAIN.AMP = True
# 
_C.TRAIN.START_EPOCH = 0
# 
_C.TRAIN.EPOCHS = 200
# 
_C.TRAIN.WARMUP_EPOCHS = 5
# 
_C.TRAIN.WEIGHT_DECAY = 1e-5
# 
_C.TRAIN.BASE_LR = 1e-3
# 
_C.TRAIN.WARMUP_LR = 1e-4
# 
_C.TRAIN.MIN_LR = 1e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = True
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Mix up parameter
_C.TRAIN.LAMBDA=0.2
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
# cosine liner step
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Early Stopping max patience
_C.TRAIN.PATIENCE = 20
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
# sgd adam adamw
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.99
# Loss
_C.TRAIN.LOSS = CN()
#'bce_focal_loss'  'ce_dice_loss'  'bce_dice_loss'
_C.TRAIN.LOSS.NAME = 'bce_dice_loss' 
# 
_C.TRAIN.LOSS.WEIGHT = 0.5
# 
_C.TRAIN.LOSS.LABELSMOOTH = 0.05
# -----------------------------------------------------------------------------
# Data augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
#
_C.AUG.MIXUP = True
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Path to output folder, overwritten by command line argument
_C.OUTPUT = 'output'
# Tag of experiment, overwritten by command line argument
_C.TAG = ''
# Frequency to save checkpoint
_C.SAVE_FREQ = 50
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 114514
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    # _update_config_from_file(config, args.cfg)

    # config.defrost()
    # if args.opts:
    #     config.merge_from_list(args.opts)

    # # merge from specific arguments
    # if args.batch_size:
    #     config.DATA.BATCH_SIZE = args.batch_size
    # if args.data_path:
    #     config.DATA.DATA_PATH = args.data_path
    # if args.resume:
    #     config.MODEL.RESUME = args.resume
    # if args.accumulation_steps:
    #     config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    # if args.use_checkpoint:
    #     config.TRAIN.USE_CHECKPOINT = True
    # if args.output:
    #     config.OUTPUT = args.output
    # if args.tag:
    #     config.TAG = args.tag
    # if args.eval:
        # config.EVAL_MODE = True

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config