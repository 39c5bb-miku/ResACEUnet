import torch
from torch import nn
import torch.nn.functional as F


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=1, stride=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        x = self.conv3(x)
        y += x
        return self.act2(y)


class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               3, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv3d(in_channels, out_channels, 1, stride=2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        x = self.conv3(x)
        y += x
        return self.act2(y)


class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=1, stride=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        x = self.conv3(x)
        y += x
        return self.act2(y)


class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x


class Seg_Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class STUNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 depth=[1, 1, 1, 1, 1, 1],
                 dims=[32, 64, 128, 256, 512, 512],
                 ):
        super().__init__()
        self.stem = nn.Sequential(
            Stem(in_channels, dims[0]),
            *[Residual_Block(dims[0], dims[0]) for _ in range(depth[0]-1)]
        )
        self.down = nn.ModuleList()
        for i in range(1, len(depth)):
            stage = nn.Sequential(
                Downsampling(dims[i-1], dims[i]),
                *[Residual_Block(dims[i], dims[i]) for _ in range(depth[i]-1)]
            )
            self.down.append(stage)
        self.up = nn.ModuleList()
        for i in range(len(depth)-1):
            stage = nn.Sequential(
                Upsampling(dims[-1-i], dims[-2-i])
            )
            self.up.append(stage)
        self.res = nn.ModuleList()
        for i in range(len(depth)-1):
            stage = nn.Sequential(
                Residual_Block(dims[-2-i]*2, dims[-2-i]),
                *[Residual_Block(dims[-2-i], dims[-2-i]) for _ in range(depth[-2-i]-1)]
            )
            self.res.append(stage)
        self.seg_head = Seg_Head(dims[0], out_channels)

    def forward(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for i in range(len(self.down)-1):
            x = self.down[i](x)
            skips.append(x)
        x = self.down[-1](x)
        for i in range(len(self.up)):
            x = self.up[i](x)
            x = torch.cat([x, skips[-1-i]], dim=1)
            x = self.res[i](x)
        x = self.seg_head(x)
        return x


if __name__ == "__main__":
    x = torch.rand(1, 1, 64, 64, 64)
    model = STUNet(1, 1)
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    output = model(x)
    print(output.shape)
