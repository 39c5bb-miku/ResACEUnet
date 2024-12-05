import os
import torch
import numpy as np
from models.build import build_model
from data.build_data import get_test_loader
from utils.conf import get_conf
from utils.sliding_window import sliding_window_inference
from collections import OrderedDict

def predict_3d(config):
    input_folder = r"/home/zph/data/self-supervised/datasets/test/seismic"
    output_folder = r'datasets/test/fault'
    overlap_rate = 0.5 #TODO
    boundary_mode = 'gaussian'
    os.makedirs(input_folder, exist_ok=True)
    input_patch_size = config.data.img_size
    resample = False
    DEVICE = 'cuda:7'
    model = build_model(config)
    newdict = OrderedDict()
    model_path = r'output/checkpoints'
    model_dict = torch.load(model_path + config.model.name + '.pt',map_location='cpu')
    for k,v in model_dict.items():
        k=k.replace('module.','')
        newdict[k]=v
    model.load_state_dict(newdict)
    if DEVICE!= 'cpu' and torch.cuda.is_available():
        print('GPU inference')
        torch.backends.cudnn.benchmark = False
    else:
        print('CPU inference')
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        data_loader=get_test_loader(input_folder)
        for data in data_loader:
            test_input = data['image'].to(DEVICE)
            filename = data['image_meta_dict']['filename_or_obj'][0].split("/")[-1]
            B,C,H,W,D=test_input.shape
            if resample:
                test_input = torch.nn.functional.interpolate(input=test_input,size=(H,W//2), mode='trilinear',align_corners=True)
            test_output = sliding_window_inference(test_input,(input_patch_size,input_patch_size,input_patch_size)
                                                , 1, model, overlap=overlap_rate, mode=boundary_mode, device=DEVICE).cpu()
            if resample:
                test_output = torch.nn.functional.interpolate(input=test_output, size=(H, W), mode='trilinear',align_corners=True)
            test_output = np.squeeze(test_output.numpy())
            if resample:
                np.save(output_folder+'/' +filename+'_resample',test_output)
            else:
                np.save(output_folder+'/'+filename,test_output)
            print('done '+filename)

if __name__ == '__main__':
    config_path = 'configs/config.yaml'
    config, _ = get_conf(config_path)
    predict_3d(config)
