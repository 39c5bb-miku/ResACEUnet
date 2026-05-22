import os
import json
import torch
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from monai.data.decathlon_datalist import load_decathlon_datalist
from monai.data.dataset import Dataset
from monai.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.transforms.croppad.dictionary import RandCropByPosNegLabeld
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianNoised,
)
from monai.transforms.spatial.dictionary import RandRotate90d, RandFlipd, RandRotated
from util.distributed import is_dist_avail_and_initialized, get_rank


def get_ddp_generator(seed=3407):
    local_rank = get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


def build_loader(config):
    g = get_ddp_generator()
    datalist_json = os.path.join(config.data.json_path, config.data.json_name)
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True),
            RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.9, 1.1)),
            RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(0, 1)),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=[0, 1]),
            RandRotated(
                keys=["image", "label"],
                range_x=0.25,
                range_y=0.25,
                range_z=0.0,
                mode="bilinear",
                prob=0.2,
                padding_mode="zeros",
            ),
            RandGaussianSmoothd(keys=["image"], prob=0.1),
            RandGaussianNoised(keys=["image"], std=0.03, prob=0.2),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                spatial_size=(
                    config.data.img_size,
                    config.data.img_size,
                    config.data.img_size,
                ),
                label_key="label",
                image_key="image",
                image_threshold=0.0,
            ),
            EnsureTyped(keys=["image", "label"], data_type="tensor"),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                spatial_size=(
                    config.data.img_size,
                    config.data.img_size,
                    config.data.img_size,
                ),
                label_key="label",
                image_key="image",
                image_threshold=0.0,
            ),
            EnsureTyped(keys=["image", "label"], data_type="tensor"),
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, "train")
    train_ds = Dataset(data=datalist, transform=train_transform)

    is_distributed = is_dist_avail_and_initialized()
    train_sampler = (
        DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=True,
        generator=g,
    )

    val_files = load_decathlon_datalist(datalist_json, True, "val")
    val_ds = Dataset(data=val_files, transform=val_transform)

    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=True,
    )

    return train_loader, val_loader


def create_json_files(foldername):
    abs_imagepath = os.path.abspath(foldername)

    images = os.listdir(abs_imagepath)
    res = {"test": []}
    template = {"image": ""}

    for image in images:
        temp = template.copy()
        temp["image"] = os.path.join(abs_imagepath, image)
        res["test"].append(temp)

    os.makedirs("data/test", exist_ok=True)
    with open("data/test/dataset.json", "w") as f:
        f.write(json.dumps(res))

    return "data/test/dataset.json"


def get_test_loader(json_dir):
    datalist_json = create_json_files(json_dir)
    test_transform = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True),
            EnsureTyped(keys=["image"], data_type="tensor"),
        ]
    )
    test_files = load_decathlon_datalist(datalist_json, True, "test")
    test_ds = Dataset(data=test_files, transform=test_transform)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        sampler=None,
        pin_memory=False,
        persistent_workers=True,
    )
    return test_loader
