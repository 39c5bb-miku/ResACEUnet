import os
import torch
from torch.amp import autocast
import numpy as np
from models.build import build_model
from data.build_data import get_test_loader
from utils.config import get_config
from monai.inferers import SlidingWindowInferer
from collections import OrderedDict

def predict_3d(config):

    input_folder = "datasets/test/seismic"
    output_folder = 'datasets/test/fault'
    os.makedirs(output_folder, exist_ok=True)
    window_infer = SlidingWindowInferer(
        roi_size=[config.data.img_size,config.data.img_size,config.data.img_size],
        sw_batch_size=2,
        overlap=0.5,
        mode='gaussian',
        progress=True,
        device=torch.device('cpu'),
    )
    model = build_model(config)
    newdict = OrderedDict()
    model_path = os.path.join('output',config.model.name,'checkpoints')
    model_dict = torch.load(os.path.join(model_path, config.model.name + '.pt'),map_location='cpu',weights_only=False)
    for k,v in model_dict.items():
        k=k.replace('module.','')
        newdict[k]=v
    model.load_state_dict(newdict)
    model.cuda()
    model.eval()
    with autocast(device_type='cuda',enabled=config.train.amp):
        with torch.no_grad():
            data_loader=get_test_loader(input_folder)
            for data in data_loader:
                test_input = data['image'].cuda()
                filename = data['image_meta_dict']['filename_or_obj'][0].split("/")[-1]
                test_output = window_infer(inputs=test_input, network=model)
                test_output = np.squeeze(torch.sigmoid(test_output[0]).numpy())
                np.save(os.path.join(output_folder,filename),test_output)
                print('done '+filename)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config_path = 'configs/config.yaml'
    config, _ = get_config(config_path)
    predict_3d(config)
