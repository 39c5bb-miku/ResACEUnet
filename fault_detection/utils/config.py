import os
import torch
from omegaconf import OmegaConf

def get_config(path):
    config = OmegaConf.load(path)
    config.gpu_num = torch.cuda.device_count()
    overrides = OmegaConf.from_cli()
    config = OmegaConf.merge(config, overrides)
    output_dir = os.path.join(config.train.output,config.model.name)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.yaml")
    with open(path, 'w') as f:
        OmegaConf.save(config, f)
    return config