import os
import sys
import argparse
import torch
from omegaconf import OmegaConf

def get_conf(path):
    if len(sys.argv) >= 2:
        conf_file = sys.argv[1]
    else:
        conf_file = path
    assert os.path.exists(conf_file), f"Config file {conf_file} does not exist!"
    conf = OmegaConf.load(conf_file)

    parser = argparse.ArgumentParser(description='SSL pre-training')
    for key in conf:
        if conf[key] is None:
            parser.add_argument(f'--{key}', default=None)
        else:
            if key == 'gpu_num':
                parser.add_argument('--gpu_num', type=int, default=torch.cuda.device_count())
            else:
                parser.add_argument(f'--{key}', type=type(conf[key]), default=conf[key])
    args = parser.parse_args()
    args = vars(args)
    args = OmegaConf.create(args)
    args_dict = OmegaConf.to_container(args)

    return args,args_dict