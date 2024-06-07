from ResACEUnet import ResACEUnet
from Swin_UnetR import Swin_UnetR
from ResUnet import ResUnet
from Unet import Unet


def build_model(config):
    model_type = config.MODEL.NAME
    if model_type == 'RESACEUNET':
        model = ResACEUnet(in_channels=config.MODEL.IN_CHANS, out_channels=config.MODEL.NUM_CLASSES,
                    img_size=(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE),#输入模型的图像尺寸
                    feature_size=16,
                    num_heads=4,
                    hidden_size=512,#隐藏层大小:隐藏层是指除了输入层和输出层之外的所有层
                    dropout_rate=config.MODEL.DROP_RATE,#dropout的比例
                    depths=[3, 3, 3],#每个卷积层的数量
                    dims=[32, 64, 512],)#每个卷积层的输出通道数
    elif model_type == 'RESUNET':
        model = ResUnet(n_channels=config.MODEL.IN_CHANS, n_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'UNET':
        model = Unet(n_channels=config.MODEL.IN_CHANS, n_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'SWIN_UNETR':
        model = Swin_UnetR(in_channels=config.MODEL.IN_CHANS, out_channels=config.MODEL.NUM_CLASSES,
                    img_size=(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE),#输入模型的图像尺寸
                    feature_size=48,
                    depths=[2, 2, 2, 2],#每个卷积层的数量
                    num_heads=[3, 6, 12, 24],
                    drop_rate=config.MODEL.DROP_RATE,#指定应在网络的输入和输出之间添加的丢失率：在训练期间，丢失率将被应用于网络的输出，以增强其泛化能力和稳健性
                    dropout_path_rate = config.MODEL.DROP_PATH_RATE)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
