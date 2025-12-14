import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.base._Embedding import timestep_embedding, unified_pos_embedding
import numpy as np
import torch.nn.functional as F


################################################################
# Multiscale modules 1D
################################################################

class DoubleConv1D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, normtype='bn'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if normtype == 'bn':
            self.double_conv = nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif normtype == 'in':
            self.double_conv = nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm1d(mid_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm1d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down1D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, normtype='bn'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels, normtype=normtype)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up1D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, normtype='bn'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv1D(in_channels, out_channels, in_channels // 2, normtype=normtype)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv1D(in_channels, out_channels, normtype=normtype)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


################################################################
# Multiscale modules 2D
################################################################
class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, normtype='bn'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if normtype == 'bn':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif normtype == 'in':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(mid_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down2D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, normtype='bn'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2D(in_channels, out_channels, normtype=normtype)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, normtype='bn'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2D(in_channels, out_channels, in_channels // 2, normtype=normtype)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2D(in_channels, out_channels, normtype=normtype)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


################################################################
# Multiscale modules 3D
################################################################

class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, normtype='bn'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if normtype == 'bn':
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif normtype == 'in':
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(mid_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, normtype='bn'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, normtype=normtype)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, normtype='bn'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2, normtype=normtype)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels, normtype=normtype)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)