from .UNet3 import UNet3
from .ResUNet import ResUNet
from .ResACEUNet import ResACEUNet2
from .SwinUNETR import SwinUNETR


def build_model(config):
    model_type = config.model.name
    if model_type == 'RESACEUNET':
        model = ResACEUNet2(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
            img_size=config.data.img_size,
            drop_rate=config.model.drop,
            attn_drop_rate=config.model.attn_drop,
        )
    elif model_type == 'RESUNET':
        model = ResUNet(
            n_channels=config.model.in_chans,
            n_classes=config.model.num_classes
    )
    elif model_type == 'UNET3':
        model = UNet3(
            n_channels=config.model.in_chans,
            n_classes=config.model.num_classes
    )
    elif model_type == 'SWIN_UNETR':
        model = SwinUNETR(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
            img_size=config.data.img_size,
            feature_size=48,
            drop_rate=config.model.drop,
            attn_drop_rate = config.model.attn_drop,
            dropout_path_rate = config.model.drop_path
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
