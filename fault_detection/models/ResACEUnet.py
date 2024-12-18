from typing import Tuple, Union, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import UnetOutBlock, UnetResBlock, get_conv_layer
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import, ensure_tuple_rep
from monai.networks.layers import trunc_normal_

class ResACEUNet2(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: Tuple[int, int, int],
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            depths=[3, 3, 3, 3],
            dims=[32, 64, 128, 256],
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            drop_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
        Examples::
            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
        """
        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.num_classes = out_channels
        if not (0 <= drop_rate <= 1):
            raise AssertionError("drop_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.feat_size = ensure_tuple_rep(img_size // 32,3)
        self.hidden_size = hidden_size

        self.ace_encoder = ACEEncoder(
            input_size=[(img_size//4)**3,(img_size//8)**3,(img_size//16)**3,(img_size//32)**3],
            dims=dims,proj_size =[64,64,64,32],
            depths=depths,
            num_heads=num_heads,
            spatial_dims=3,
            in_channels=in_channels,
            dropout=drop_rate,
            attn_drop_rate=attn_drop_rate
        )

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            dropout=drop_rate
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=(img_size//16)**3,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=(img_size//8)**3,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=(img_size//4)**3,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=4,
            norm_name=norm_name,
            out_size=(img_size)**3,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        hidden_states = self.ace_encoder(x_in)
        convBlock = self.encoder1(x_in)
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        out = self.decoder2(dec1, convBlock)
        logits = self.out1(out)

        return logits

einops, _ = optional_import("einops")

class ACEEncoder(nn.Module):

    def __init__(self, input_size, dims, proj_size, depths, num_heads, spatial_dims, in_channels, dropout, attn_drop_rate):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(4, 4, 4), stride=(4, 4, 4),
                        dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                            dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],  proj_size=proj_size[i], num_heads=num_heads, drop_rate=attn_drop_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))

        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return hidden_states

    def forward(self, x):
        hidden_states = self.forward_features(x)
        return hidden_states


class TransformerBlock(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            drop_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            drop_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.
        """
        super().__init__()
        if not (0 <= drop_rate <= 1):
            raise ValueError("drop_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.ace_block = ACE(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=drop_rate,spatial_attn_drop=drop_rate,proj_drop=drop_rate)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.2, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.ace_block(self.norm(x).permute(0, 2, 1).reshape(B, C, H, W, D))

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


class ACE(nn.Module):

    def __init__(self, input_size, hidden_size, proj_size,
                channel_attn_drop, spatial_attn_drop, proj_drop, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)
        self.fc = nn.Conv2d(4*self.num_heads, 16, kernel_size=1, bias=True)
        self.dep_conv = nn.Conv3d(16*hidden_size//self.num_heads, hidden_size, kernel_size=3, bias=True, groups=hidden_size//self.num_heads, padding=1)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.rate = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W, D = x.shape

        qkvv = self.qkvv(x.permute(0, 2, 3, 4, 1).reshape(B, H * W * D, C))

        # fully connected layer
        f_all = qkvv.reshape(B, H * W * D, 4*self.num_heads, -1).permute(0, 2, 1, 3) # B, 4*nhead, H*W*D, C//nhead
        f_conv = self.fc(f_all).permute(0, 3, 1, 2).reshape(B, 16*x.shape[1]//self.num_heads, H, W, D) # B, 16*C//nhead, H, W, D
        # group conovlution
        out_conv = self.dep_conv(f_conv).permute(0, 2, 3, 4, 1).reshape(B, H * W * D, C)

        qkvv = qkvv.reshape(B, H * W * D, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = F.normalize(q_shared, dim=-1)
        k_shared = F.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, H * W * D, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, H * W * D, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x_attention = torch.cat((x_SA, x_CA), dim=-1)
        x = self.rate * x_attention + self.rate2 * out_conv
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2', 'rate', 'rate2'}


class UnetrUpBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                            norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,drop_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
