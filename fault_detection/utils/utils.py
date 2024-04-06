import os
import torch
import math
import torch.nn.functional as F

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir,logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def ensure_match_dict(model_dict,new_dict):
    for k, dst_weight in model_dict.items():
        src_weight = new_dict[k]
        if dst_weight.shape != src_weight.shape:
            if 'pos_embed' in k:
                Hh = Wh = int(math.sqrt(src_weight.shape[1]))
                H = W = int(math.sqrt(dst_weight.shape[1]))
                C = dst_weight.shape[2]
                src_weight = src_weight.reshape(1, Hh, Wh, C).permute(0, 3, 1, 2)
                src_weight = F.interpolate(src_weight, size=(H, W), mode='bicubic')
                src_weight = torch.flatten(src_weight, 2).transpose(1, 2)
            elif 'EF' in k:
                Hh = Wh = int(math.sqrt(src_weight.shape[0]))
                H = W = int(math.sqrt(dst_weight.shape[0]))
                C = dst_weight.shape[1]
                src_weight = src_weight.reshape(1, Hh, Wh, C).permute(0, 3, 1, 2)
                src_weight = F.interpolate(src_weight, size=(H, W), mode='bicubic')
                src_weight = torch.flatten(src_weight, 2).transpose(1, 2)
                src_weight = torch.squeeze(src_weight)
        new_dict[k] = src_weight
    return new_dict


def find_ckptpath(config):
    ckpt_path = config.OUTPUT
    ckpt_name = config.CKPT
    total_path = os.path.join(ckpt_path,ckpt_name)
    best_ckpt_path = os.path.join(ckpt_path,'best',ckpt_name)
    if(os.path.exists(total_path)):
        return total_path
    elif(os.path.exists(best_ckpt_path)):
        return best_ckpt_path
    elif(os.path.exists(ckpt_name)):
        return ckpt_name
    else:
        raise ValueError("Please check your path.")


def find_filename_in_path(path):
    filename = ''
    for i in range(len(path)-1,-1,-1):
        if(path[i]=="\\" or path[i]=='/'):
            break
        else:
            filename = path[i] + filename
    return filename


def check_dataset_exist(path):
    if os.path.exists(path+'/'+'train') and os.path.exists(path+'/'+'val'):
        if os.path.exists(path+'/'+'train/image'):
            path=os.listdir(path+'/'+'train/image')
            for file in path:
                if '.npy' in file:
                    return True
    return False






