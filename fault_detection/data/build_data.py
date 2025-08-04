import os
import json
import torch
from monai.transforms import *
from monai.data import load_decathlon_datalist,Dataset,DataLoader


def build_loader(config):
    data_dir = config.data.json_path
    datalist_json = os.path.join(data_dir, config.data.json_name)
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"],image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            EnsureChannelFirstd(keys=["label"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.9,1.1)),
            RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(0,1)),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=[0,1]),
            RandRotated(keys=["image", "label"], range_x=0.25, range_y=0.25, range_z=0.0, mode="bilinear",prob=0.2, padding_mode="zeros"),
            RandGaussianSmoothd(keys=["image"], prob=0.1),
            RandGaussianNoised(keys=["image"], std=0.03, prob=0.2),
            RandCropByPosNegLabeld(keys=["image", "label"], spatial_size=(config.data.img_size,config.data.img_size,config.data.img_size),label_key="label",image_key="image",image_threshold=0.), 
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"],image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            EnsureChannelFirstd(keys=["label"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            RandCropByPosNegLabeld(keys=["image", "label"], spatial_size=(config.data.img_size,config.data.img_size,config.data.img_size),label_key="label",image_key="image",image_threshold=0.), 
            ToTensord(keys=["image", "label"]),
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, "training")
    train_ds = Dataset(data=datalist, transform=train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle = True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=False,
    )
    val_files = load_decathlon_datalist(datalist_json, True, "validation")
    val_ds = Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=False,
    )

    return train_loader, val_loader


def create_json_files(imagepath):
    images = os.listdir(imagepath)
    res = {"testing":[]}
    template = {"image":""}
    for image in images:
        temp = template.copy()
        temp["image"] = imagepath+'/'+ image
        res["testing"].append(temp)
    with open("datasets/test/datas.json", 'w') as f:
        f.write(json.dumps(res))
    return "datasets/test/datas.json"


def get_test_loader(json_dir):
    datalist_json = create_json_files(json_dir)
    test_transform = Compose(
        [
            LoadImaged(keys=["image"],image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            ToTensord(keys=["image"]),
        ]
    )
    test_files = load_decathlon_datalist(datalist_json, True, "testing")
    test_ds = Dataset(data=test_files, transform=test_transform) 
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False,
    )
    return test_loader
