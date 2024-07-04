import os
import json
import random
import torch
import torch.distributed as dist
from monai.transforms import *
from monai.data import load_decathlon_datalist,Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_ddp_generator(seed=3407):

    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


def build_loader(config):
    data_dir = config.data.json_path#数据集的json文件所在的路径
    datalist_json = os.path.join(data_dir, config.data.json_name)#数据集的json文件
    scale_list = [160, 192, 224]
    space_list = [128, 160]
    rand_size = random.choice(scale_list)
    spatial_rand_size = random.choice(space_list)
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],#从给定的图像中随机裁剪具有积极和/或负面标签的体积块，keys：要处理的数据的名称
                label_key="label",#label_key：标签数据的名称
                spatial_size=(config.data.img_size,config.data.img_size,config.data.img_size),#spatial_size：所提取的样本的空间大小
                pos=1,#pos：表示从标签图中选择的正例样本数
                neg=1,#neg：表示从标签图中选择的负例样本数
                num_samples=1, #num_samples：表示从输入图像中选择的样本数
                image_key="image",#image_key：表示输入数据字典中的键，对应输入图像数据
                image_threshold=0,#image_threshold：对于二进制分割任务，表示在二值化阈值下的阈值值，像素值大于该值则被视为正标签，小于等于该值则被视为负标签
            ),
            Resized(keys=["image", "label"], spatial_size=(spatial_rand_size,spatial_rand_size,rand_size), mode='trilinear',align_corners=False),
            RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.9,1.1)),
            RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(0,1)),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=[0,1]),
            RandRotated(keys=["image", "label"], range_x=0.25, range_y=0.25, range_z=0.0, mode="bilinear",prob=0.6, padding_mode="zeros"),
            RandGaussianSmoothd(keys=["image"], prob=0.1),
            RandGaussianNoised(keys=["image"], std=0.03, prob=0.2),
            CenterSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128)),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),#将给定的Nifti格式的图像和标签数据加载到内存中，然后将其存储在字典数据结构中
            EnsureChannelFirstd(keys=["image", "label"]),#将图像和标签添加到通道维度
            RandCropByPosNegLabeld(
                keys=["image", "label"],#从给定的图像中随机裁剪具有积极和/或负面标签的体积块，keys：要处理的数据的名称
                label_key="label",#label_key：标签数据的名称
                spatial_size=(config.data.img_size,config.data.img_size,config.data.img_size),#spatial_size：所提取的样本的空间大小
                pos=1,#pos：表示从标签图中选择的正例样本数
                neg=1,#neg：表示从标签图中选择的负例样本数
                num_samples=1, #num_samples：表示从输入图像中选择的样本数
                image_key="image",#image_key：表示输入数据字典中的键，对应输入图像数据
                image_threshold=0,#image_threshold：对于二进制分割任务，表示在二值化阈值下的阈值值，像素值大于该值则被视为正标签，小于等于该值则被视为负标签
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),#将图像的强度范围缩放到给定的最小值和最大值之间
            ToTensord(keys=["image", "label"]),#将图像和标签转换为PyTorch张量
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, "training")#从一个JSON文件中加载数据列表
    train_ds = Dataset(data=datalist, transform=train_transform)#根据给定的 datalist 和 train_transform 构建一个 dataset 对象，用于训练模型
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds,#加载的数据集
        batch_size=config.data.batch_size,#每个batch的大小
        shuffle = False,
        sampler = train_sampler,
        num_workers=config.data.num_workers,#加载数据的线程数
        pin_memory=config.data.pin_memory,#是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
        persistent_workers=True,#是否在子进程中保持数据集对象
    )
    val_files = load_decathlon_datalist(datalist_json, True, "validation")#从一个JSON文件中加载数据列表
    val_ds = Dataset(data=val_files, transform=val_transform)#根据给定的 datalist 和 val_transform 构建一个 dataset 对象，用于验证模型
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    val_loader = DataLoader(
        val_ds,#加载的数据集
        batch_size=1,#每个batch的大小
        shuffle=False,#是否打乱数据
        sampler=val_sampler,#采样器    
        num_workers=config.data.num_workers,#加载数据的线程数
        pin_memory=config.data.pin_memory,#是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
        persistent_workers=True,#是否在子进程中保持数据集对象
    )

    return train_loader, val_loader


def create_json_files(foldername):
    imagepath = foldername  ##totalpath=foldername+imagepath
    images = os.listdir(imagepath)
    res = {"testing":[]}
    template = {"image":""}
    for image in images:
        temp = template.copy()
        temp["image"] = imagepath+'/'+ image
        res["testing"].append(temp)
    with open('datasets/test/datas.json', 'w') as f:
        f.write(json.dumps(res))
    return 'datasets/test/datas.json'


def get_test_loader(json_dir):
    json_name = create_json_files(json_dir)
    datalist_json = json_name#os.path.join(json_dir, json_name)
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
    test_sampler = None
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        sampler=test_sampler,
        pin_memory=False,
        persistent_workers=True,
    )
    return test_loader
