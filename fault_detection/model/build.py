from .UNet3 import UNet3
from .UNet4 import UNet4
from .ResUNet import ResUNet
from .UNETR import UNETR
from .UNETR_PP import UNETR_PP
from .ResACEUNet2 import ResACEUNet2
from .SwinUNETR import SwinUNETR
from .UMambaOut import UMambaOut
from .STUNet import STUNet
from .SimAUNeXt import SimAUNeXt


def build_model(config):
    model_type = config.model.name
    if model_type == "UNETR_PP":
        model = UNETR_PP(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
            img_size=(config.data.img_size, config.data.img_size, config.data.img_size),
            dropout_rate=config.model.drop,
            attn_drop_rate=config.model.attn_drop,
            do_ds=config.model.ds,
        )
    elif model_type == "RESACEUNET2":
        model = ResACEUNet2(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
            img_size=config.data.img_size,
            drop_rate=config.model.drop,
            attn_drop_rate=config.model.attn_drop,
        )
    elif model_type == "UNETR":
        model = UNETR(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
            img_size=(config.data.img_size, config.data.img_size, config.data.img_size),
            dropout_rate=config.model.drop,
        )
    elif model_type == "RESUNET":
        model = ResUNet(
            n_channels=config.model.in_chans, n_classes=config.model.num_classes
        )
    elif model_type == "UNET3":
        model = UNet3(
            n_channels=config.model.in_chans, n_classes=config.model.num_classes
        )
    elif model_type == "UNET4":
        model = UNet4(
            n_channels=config.model.in_chans, n_classes=config.model.num_classes
        )
    elif model_type == "SWINUNETR":
        model = SwinUNETR(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
            img_size=config.data.img_size,
            feature_size=24,
            drop_rate=config.model.drop,
            attn_drop_rate=config.model.attn_drop,
            dropout_path_rate=config.model.drop_path,
        )
    elif model_type == "UMAMBAOUT":
        model = UMambaOut(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
            img_size=config.data.img_size,
            drop_rate=config.model.drop,
            attn_drop_rate=config.model.attn_drop,
            drop_path_rate=config.model.drop_path,
            do_ds=config.model.ds,
        )
    elif model_type == "STUNET":
        model = STUNet(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
        )
    elif model_type == "SIMAUNEXT":
        model = SimAUNeXt(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
            img_size=config.data.img_size,
            feature_size=32,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
