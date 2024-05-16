import os
import json
from monai import data, transforms
from monai.data import load_decathlon_datalist

def get_loader(config):
    data_dir = config.DATA.JSON_PATH#数据集的json文件所在的路径
    datalist_json = os.path.join(data_dir, config.DATA.JSON_NAME)#数据集的json文件
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),#将给定的Nifti格式的图像和标签数据加载到内存中，然后将其存储在字典数据结构中
            transforms.AddChanneld(keys=["image", "label"]),#将图像和标签添加到通道维度
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),#将图像的强度范围缩放到给定的最小值和最大值之间
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),#从图像中提取感兴趣区域，根据该区域进行裁剪
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],#从给定的图像中随机裁剪具有积极和/或负面标签的体积块，keys：要处理的数据的名称
                label_key="label",#label_key：标签数据的名称
                spatial_size=(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE),#spatial_size：所提取的样本的空间大小
                pos=1,#pos：表示从标签图中选择的正例样本数
                neg=1,#neg：表示从标签图中选择的负例样本数
                num_samples=1, #num_samples：表示从输入图像中选择的样本数
                image_key="image",#image_key：表示输入数据字典中的键，对应输入图像数据
                image_threshold=0,#image_threshold：对于二进制分割任务，表示在二值化阈值下的阈值值，像素值大于该值则被视为正标签，小于等于该值则被视为负标签
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            transforms.RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, mode=["bilinear", "nearest"],
                        prob=0.2,padding_mode="border"),
            transforms.RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15),
                                prob=0.15),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),#将给定的Nifti格式的图像和标签数据加载到内存中，然后将其存储在字典数据结构中
            transforms.AddChanneld(keys=["image", "label"]),#将图像和标签添加到通道维度
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),#将图像的强度范围缩放到给定的最小值和最大值之间
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],#从给定的图像中随机裁剪具有积极和/或负面标签的体积块，keys：要处理的数据的名称
                label_key="label",#label_key：标签数据的名称
                spatial_size=(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE),#spatial_size：所提取的样本的空间大小
                pos=1,#pos：表示从标签图中选择的正例样本数
                neg=1,#neg：表示从标签图中选择的负例样本数
                num_samples=1,#num_samples：表示从输入图像中选择的样本数
                image_key="image",#image_key：表示输入数据字典中的键，对应输入图像数据
                image_threshold=0,#image_threshold：对于二进制分割任务，表示在二值化阈值下的阈值值，像素值大于该值则被视为正标签，小于等于该值则被视为负标签
            ),
            transforms.ToTensord(keys=["image", "label"]),#将图像和标签转换为PyTorch张量
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, "training")#从一个JSON文件中加载数据列表
    train_ds = data.Dataset(data=datalist, transform=train_transform)#根据给定的 datalist 和 train_transform 构建一个 Dataset 对象，用于训练模型
    train_loader = data.DataLoader(
        train_ds,#加载的数据集
        batch_size=config.DATA.BATCH_SIZE,#每个batch的大小
        shuffle=True,#是否打乱数据
        num_workers=config.DATA.NUM_WORKERS,#加载数据的线程数
        pin_memory=config.DATA.PIN_MEMORY,#是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
        persistent_workers=True,#是否在子进程中保持数据集对象
    )
    val_files = load_decathlon_datalist(datalist_json, True, "validation")#从一个JSON文件中加载数据列表
    val_ds = data.Dataset(data=val_files, transform=val_transform)#根据给定的 datalist 和 val_transform 构建一个 Dataset 对象，用于验证模型
    val_sampler = None
    val_loader = data.DataLoader(
        val_ds,#加载的数据集
        batch_size=1,#每个batch的大小
        shuffle=False,#是否打乱数据
        num_workers=config.DATA.NUM_WORKERS,#加载数据的线程数
        sampler=val_sampler,#采样器
        pin_memory=config.DATA.PIN_MEMORY,#是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
        persistent_workers=True,#是否在子进程中保持数据集对象
    )

    return train_ds, val_ds,train_loader, val_loader


def create_json_files(foldername):
    imagepath = foldername  ##totalpath=foldername+imagepath
    images = os.listdir(imagepath)
    res = {"testing":[]}
    template = {"image":""}
    for image in images:
        temp = template.copy()
        temp["image"] = './'+imagepath+'/'+ image
        res["testing"].append(temp)
    with open('datas.json', 'w') as f:
        f.write(json.dumps(res))
    return 'datas.json'


def get_validation_loader(config):
    data_dir = config.DATA.JSON_PATH
    datalist_json = os.path.join(data_dir, config.DATA.JSON_NAME)
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),
            # transforms.CropForegroundd(keys=["image"], source_key="image"),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.ToTensord(keys=["image"]),
        ]
    )
    val_files = load_decathlon_datalist(datalist_json, True, "validation")
    val_ds = data.Dataset(data=val_files, transform=val_transform)
    val_sampler = None
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        sampler=val_sampler,
        pin_memory=config.DATA.PIN_MEMORY,
        persistent_workers=True,
    )
    return val_loader


def get_test_loader(json_dir):
    json_name = create_json_files(json_dir)
    datalist_json = json_name#os.path.join(json_dir, json_name)
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),
            # transforms.CropForegroundd(keys=["image"], source_key="image"),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.ToTensord(keys=["image"]),
        ]
    )
    test_files = load_decathlon_datalist(datalist_json, True, "testing")
    test_ds = data.Dataset(data=test_files, transform=test_transform) 
    test_sampler = None
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        sampler=test_sampler,
        pin_memory=False,
        persistent_workers=True,
    )
    return test_loader
