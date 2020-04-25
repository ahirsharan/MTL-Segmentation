""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv2d_mtl import Conv2dMtl, ConvTranspose2dMtl


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def conv3x3mtl(in_planes, out_planes, stride=1):
    return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class DoubleConvMtl(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = conv3x3mtl(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3mtl(mid_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, mtl):
        super().__init__()
        if mtl:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConvMtl(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mtl):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        if mtl:
            self.up = ConvTranspose2dMtl(in_channels , in_channels // 2, 2, 2)
            self.conv = DoubleConvMtl(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, mtl):
        super(OutConv, self).__init__()
        if mtl:
            self.conv = Conv2dMtl(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetMtl(nn.Module):
    def __init__(self, n_channels, n_classes, mtl=True):
        super(UNetMtl, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = False
        self.mtl = mtl
        if mtl:
            self.inc = DoubleConvMtl(n_channels, 64)
        else:
            self.inc = DoubleConv(n_channels, 64)
            
        self.down1 = Down(64, 128, mtl)
        self.down2 = Down(128, 256, mtl)
        self.down3 = Down(256, 512, mtl)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, mtl)
        self.up1 = Up(1024, 512, mtl)
        self.up2 = Up(512, 256, mtl)
        self.up3 = Up(256, 128, mtl)
        self.up4 = Up(128, 64 * factor, mtl)
        self.outc = Down(64, n_classes, mtl)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # print(x.size())
        logits = self.outc(x) #segmented ouput
        
        #Classifier Use and the layer changed
        #logits = logits.view(logits.size(0), -1)
        #print(logits.size())
        return logits
