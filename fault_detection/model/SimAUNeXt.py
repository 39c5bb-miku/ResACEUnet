from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.patchembedding import PatchEmbed
from monai.networks.blocks.unetr_block import UnetrBasicBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.dynunet_block import (
    UnetBasicBlock,
    UnetResBlock,
    get_conv_layer,
)
from monai.networks.layers.utils import get_norm_layer
from monai.networks.layers.weight_init import trunc_normal_
from monai.networks.layers.drop_path import DropPath
from monai.utils.module import optional_import
from monai.utils.misc import ensure_tuple_rep
from torch_dwt.functional import dwt3

rearrange, _ = optional_import("einops", name="rearrange")


class SimAUNeXt(nn.Module):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: int,
        depth: int = 3,
        num_heads: int = 8,
        feature_size: int = 32,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ape: bool = True,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            img_size=img_size,
            in_chans=in_channels,
            embed_dim=feature_size,
            depth=depth,
            num_heads=num_heads,
            norm_name=norm_name,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            spatial_dims=spatial_dims,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            upsample_kernel_size=2,
            img_size=img_size // 8,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            upsample_kernel_size=2,
            img_size=img_size // 4,
        )
        self.decoder2 = UnetUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1 = UnetUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            norm_name=norm_name,
            res_block=True,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def forward(self, x_in):
        hidden_states_out = self.encoder(x_in)
        dec3 = self.decoder4(hidden_states_out[4], hidden_states_out[3])
        dec2 = self.decoder3(dec3, hidden_states_out[2])
        dec1 = self.decoder2(dec2, hidden_states_out[1])
        out = self.decoder1(dec1, hidden_states_out[0])
        logits = self.out(out)
        return logits


class Conv(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        norm_name,
        res_block,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            # nn.MaxPool3d(2),
            Down_wt(in_channels, out_channels),
            UnetrBasicBlock(
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size,
                stride,
                norm_name,
                res_block,
            ),
        )

    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        eta: float = 1.0,
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimA(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop_rate,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(
            hidden_size=dim,
            mlp_dim=mlp_hidden_dim,
            dropout_rate=drop_rate,
            dropout_mode="swin",
        )
        self.norm3 = nn.LayerNorm(dim)
        self.local_mp = LPI(in_features=dim, drop=drop_rate)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x, H, W, D):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W, D))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class LPI(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        kernel_size=3,
    ):
        super().__init__()
        out_features = out_features or in_features
        padding = kernel_size // 2
        self.conv1 = torch.nn.Conv3d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
        )
        self.act = act_layer()
        self.bn = nn.SyncBatchNorm(in_features)
        self.conv2 = torch.nn.Conv3d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
        )

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x


class SimA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = F.normalize(k, p=1.0, dim=-2)
        q = F.normalize(q, p=1.0, dim=-2)
        if N < (C // self.num_heads):
            x = ((q @ k.transpose(-2, -1)) @ v).transpose(1, 2).reshape(B, N, C)
        else:
            x = (q @ (k.transpose(-2, -1) @ v)).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.unused
    def no_weight_decay(self):
        return {}


class BasicLayer(nn.Module):
    def __init__(
        self,
        img_size: int,
        dim: int,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        pos_embed: bool = True,
    ) -> None:
        super().__init__()
        input_size = img_size**3
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop,
                    attn_drop,
                    drop_path,
                )
                for i in range(depth)
            ]
        )
        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, dim))

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x, H, W, D)
        x = x.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.reduction = nn.Linear(input_channels * 8, input_channels * 2, bias=False)
        self.norm = nn.LayerNorm(input_channels * 8)

    def forward(self, x):
        b, c, h, w, d = x.shape
        x = x.view(b, c, h // 2, 2, w // 2, 2, d // 2, 2)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(b, h // 2, w // 2, d // 2, -1)
        x = self.norm(x)
        x = self.reduction(x).permute(0, 4, 1, 2, 3)
        return x


# class PatchMerging(nn.Module):
#     def __init__(self, input_channels):
#         super().__init__()
#         self.reduction = nn.Conv3d(input_channels * 8, input_channels * 2, 1)
#         self.norm = nn.BatchNorm3d(input_channels * 2)
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         b, c, h, w, d = x.shape
#         x = x.view(b, c, h // 2, 2, w // 2, 2, d // 2, 2)
#         x = x.reshape(b, -1, h // 2, w // 2, d // 2)
#         x = self.reduction(x)
#         x = self.norm(x)
#         x = self.relu(x)
#         return x


class Vit(nn.Module):
    def __init__(
        self,
        img_size,
        dim,
        depth,
        num_heads,
        mlp_ratio,
        qkv_bias,
        drop,
        attn_drop,
        drop_path,
        pos_embed,
    ):
        super().__init__()
        self.vit = nn.Sequential(
            PatchMerging(dim),
            BasicLayer(
                img_size,
                dim * 2,
                depth,
                num_heads,
                mlp_ratio,
                qkv_bias,
                drop,
                attn_drop,
                drop_path,
                pos_embed,
            ),
        )

    def forward(self, x):
        return self.vit(x)


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv3d(in_ch * 8, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = dwt3(x, "haar")
        B, _, C, H, W, D = x.shape
        x = x.view(B, 8 * C, H, W, D)
        x = self.conv_bn_relu(x)
        return x


class UnetUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode="nearest")
        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                in_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(
                spatial_dims,
                in_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        out = self.up(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetrUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        upsample_kernel_size: Sequence[int] | int,
        img_size: int,
    ) -> None:
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
        self.vitblock = BasicLayer(img_size=img_size, dim=out_channels, pos_embed=True)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = out + skip
        out = self.vitblock(out)
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        in_chans: int,
        embed_dim: int,
        depth: int = 3,
        num_heads: int = 8,
        norm_name: tuple | str = "instance",
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ape: bool = False,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ape = ape
        self.stem = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        patches_resolution = [img_size // 4, img_size // 4, img_size // 4]
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim * 4, *patches_resolution)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        for i_layer in range(4):
            conv = Conv(
                spatial_dims=spatial_dims,
                in_channels=int(embed_dim * 2**i_layer),
                out_channels=int(embed_dim * 2 ** (i_layer + 1)),
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )
            vit = Vit(
                img_size=int(img_size // (2 ** (i_layer + 1))),
                dim=int(embed_dim * 2**i_layer),
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                pos_embed=True,
            )
            if i_layer == 0:
                self.layers1.append(conv)
            elif i_layer == 1:
                self.layers2.append(conv)
            elif i_layer == 2:
                self.layers3.append(vit)
            elif i_layer == 3:
                self.layers4.append(vit)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layers1[0](x0.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        if self.ape:
            x2 = x2 + self.absolute_pos_embed
        x3 = self.layers3[0](x2.contiguous())
        x4 = self.layers4[0](x3.contiguous())
        return [x0, x1, x2, x3, x4]


if __name__ == "__main__":
    input = torch.rand((1, 1, 64, 64, 64))
    model = SimAUNeXt(
        in_channels=1,
        out_channels=1,
        img_size=64,
        spatial_dims=3,
        feature_size=32,
    )
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))
    output = model(input)
    print(output.shape)
