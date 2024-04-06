from ResACEUnet import ResACEUnet
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
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
