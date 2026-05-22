import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, mid_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, out_channels,
                               kernel_size=3, padding=1)
        self.shortcut_conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, stride=stride)
        self.shortcut_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut_conv(x))
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + shortcut


class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.shortcut_conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=1)
        self.shortcut_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        shortcut = self.shortcut_bn(self.shortcut_conv(x))
        return out + shortcut


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ResBlock(in_channels, out_channels, stride=2)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 2, 2),
                              mode='trilinear', align_corners=True)
        self.conv = ResBlock(in_channels, out_channels, stride=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = InitialBlock(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.up2 = Up(384, 128)
        self.up3 = Up(192, 64)
        self.up4 = Up(96, 32)
        self.out = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)

        return logits


if __name__ == "__main__":
    input = torch.randint(
        low=0,
        high=255,
        size=(1, 1, 64, 64, 64),
        dtype=torch.float,
    )
    inchannel = 1
    outchannel = 1
    model = ResUNet(n_channels=inchannel, n_classes=outchannel)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    output = model(input)
    print(output.shape)
