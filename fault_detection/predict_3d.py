import os
import torch
from torch.amp.autocast_mode import autocast
import numpy as np
from model.build import build_model
from data.build_data import get_test_loader
from util.config import get_config
from monai.inferers.inferer import SlidingWindowInferer
from collections import OrderedDict


def pred_step(model, samples, window_infer, config):
    with autocast(device_type="cuda", enabled=config.train.amp):
        outputs = window_infer(inputs=samples, network=model)
    return outputs


@torch.no_grad()
def pred(config):

    input_folder = "data/test/seismic"
    output_folder = "data/test/fault"
    os.makedirs(output_folder, exist_ok=True)
    window_infer = SlidingWindowInferer(
        roi_size=[config.data.img_size, config.data.img_size, config.data.img_size],
        sw_batch_size=1,
        overlap=0.5,
        mode="gaussian",
        progress=True,
        device=torch.device("cpu"),
    )
    model = build_model(config)
    newdict = OrderedDict()
    model_path = os.path.join("output", config.model.name, "best_model")
    model_dict = torch.load(
        os.path.join(model_path, config.model.name + ".pt"),
        map_location="cpu",
        weights_only=False,
    )
    for k, v in model_dict.items():
        k = k.replace("module.", "")
        newdict[k] = v
    model.load_state_dict(newdict)
    model.cuda()
    model.eval()
    data_loader = get_test_loader(input_folder)
    for data in data_loader:
        test_input = data["image"].cuda()
        meta_dict = data.get("image_meta_dict", data["image"].meta)
        filename = meta_dict["filename_or_obj"][0].split("/")[-1]
        test_output = pred_step(model, test_input, window_infer, config)
        test_output = torch.sigmoid(test_output[0]).detach().cpu().numpy()
        test_output = np.squeeze(test_output)
        np.save(os.path.join(output_folder, filename), test_output)
        print("done " + filename)


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    config = get_config(config_path)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu[0])
    pred(config)
