from .Unet import Unet
from .ResUnet import ResUnet
from .ResACEUnet import ResACEUnet
from .Swin_UnetR import Swin_UnetR


def build_model(config):
    model_type = config.model.name
    if model_type == 'RESACEUNET':
        model = ResACEUnet(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
            img_size=(config.data.img_size,config.data.img_size,config.data.img_size),
            feature_size=16,
            num_heads=4,
            hidden_size=512,
            dropout_rate=config.model.drop,
            attn_drop_rate=config.model.attn_drop,
            depths=[3, 3, 3],
            dims=[32, 64, 512],
            do_ds=config.model.ds
    )
    elif model_type == 'RESUNET':
        model = ResUnet(
            n_channels=config.model.in_chans,
            n_classes=config.model.num_classes
    )
    elif model_type == 'UNET':
        model = Unet(
            n_channels=config.model.in_chans,
            n_classes=config.model.num_classes
    )
    elif model_type == 'SWIN_UNETR':
        model = Swin_UnetR(
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
