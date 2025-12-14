import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from models.base._Basic import MLP
from models.base._Embedding import timestep_embedding, unified_pos_embedding
from models.base._UNet_Blocks import DoubleConv1D, Down1D, Up1D, OutConv1D, DoubleConv2D, Down2D, Up2D, OutConv2D, \
    DoubleConv3D, Down3D, Up3D, OutConv3D

ConvList = [None, DoubleConv1D, DoubleConv2D, DoubleConv3D]
DownList = [None, Down1D, Down2D, Down3D]
UpList = [None, Up1D, Up2D, Up3D]
OutList = [None, OutConv1D, OutConv2D, OutConv3D]


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    

class Model(nn.Module):
    def __init__(self, args, bilinear=True):
        super(Model, self).__init__()
        self.__name__ = 'UNOv2'
        self.args = args
        normtype = 'in'  
        shapelist = args.shape_list
        n_hidden = args.width
        act=args.act
        fun_dim = args.in_dim
        out_dim = args.out_dim
        modes = args.modes
        ## embedding
        self.preprocess = MLP(fun_dim, n_hidden * 2,
                                n_hidden, n_layers=0, res=False, act=act)

        patch_size = [(size + (16 - size % 16) % 16) // 16 for size in shapelist]
        self.padding = [(16 - size % 16) % 16 for size in shapelist]
        self.augmented_resolution = [shape + padding for shape, padding in zip(shapelist, self.padding)]

        # multiscale modules
        self.inc = ConvList[len(patch_size)](n_hidden, n_hidden, normtype=normtype)
        self.down1 = DownList[len(patch_size)](n_hidden, n_hidden * 2, normtype=normtype)
        self.down2 = DownList[len(patch_size)](n_hidden * 2, n_hidden * 4, normtype=normtype)
        self.down3 = DownList[len(patch_size)](n_hidden * 4, n_hidden * 8, normtype=normtype)
        factor = 2 if bilinear else 1
        self.down4 = DownList[len(patch_size)](n_hidden * 8, n_hidden * 16 // factor, normtype=normtype)
        self.up1 = UpList[len(patch_size)](n_hidden * 16, n_hidden * 8 // factor, bilinear, normtype=normtype)
        self.up2 = UpList[len(patch_size)](n_hidden * 8, n_hidden * 4 // factor, bilinear, normtype=normtype)
        self.up3 = UpList[len(patch_size)](n_hidden * 4, n_hidden * 2 // factor, bilinear, normtype=normtype)
        self.up4 = UpList[len(patch_size)](n_hidden * 2, n_hidden, bilinear, normtype=normtype)
        self.outc = OutList[len(patch_size)](n_hidden, n_hidden)
        # Down FNO
        self.process1_down = SpectralConv2d(n_hidden, n_hidden,
                                                        *[max(1, min(modes, min(self.augmented_resolution) // 2))
                                                          for _ in
                                                          range(len(self.padding))])
        self.process2_down = SpectralConv2d(n_hidden * 2, n_hidden * 2,
                                                        *[max(1, min(modes, min(self.augmented_resolution) // 4))
                                                          for _ in
                                                          range(len(self.padding))])
        self.process3_down = SpectralConv2d(n_hidden * 4, n_hidden * 4,
                                                        *[max(1, min(modes, min(self.augmented_resolution) // 8))
                                                          for _ in
                                                          range(len(self.padding))])
        self.process4_down = SpectralConv2d(n_hidden * 8, n_hidden * 8,
                                                        *[max(1, min(modes, min(self.augmented_resolution) // 16))
                                                          for _ in
                                                          range(len(self.padding))])
        self.process5_down = SpectralConv2d(n_hidden * 16 // factor, n_hidden * 16 // factor,
                                                        *[max(1, min(modes, min(self.augmented_resolution) // 32))
                                                          for _ in
                                                          range(len(self.padding))])
        self.w1_down = ConvList[len(self.padding)](n_hidden, n_hidden, 1)
        self.w2_down = ConvList[len(self.padding)](n_hidden * 2, n_hidden * 2, 1)
        self.w3_down = ConvList[len(self.padding)](n_hidden * 4, n_hidden * 4, 1)
        self.w4_down = ConvList[len(self.padding)](n_hidden * 8, n_hidden * 8, 1)
        self.w5_down = ConvList[len(self.padding)](n_hidden * 16 // factor, n_hidden * 16 // factor, 1)
        # Up FNO
        self.process1_up = SpectralConv2d(n_hidden, n_hidden,
                                                      *[max(1, min(modes, min(self.augmented_resolution) // 2)) for
                                                        _ in
                                                        range(len(self.padding))])
        self.process2_up = SpectralConv2d(n_hidden * 2 // factor, n_hidden * 2 // factor,
                                                      *[max(1, min(modes, min(self.augmented_resolution) // 4)) for
                                                        _ in
                                                        range(len(self.padding))])
        self.process3_up = SpectralConv2d(n_hidden * 4 // factor, n_hidden * 4 // factor,
                                                      *[max(1, min(modes, min(self.augmented_resolution) // 8)) for
                                                        _ in
                                                        range(len(self.padding))])
        self.process4_up = SpectralConv2d(n_hidden * 8 // factor, n_hidden * 8 // factor,
                                                      *[max(1, min(modes, min(self.augmented_resolution) // 16))
                                                        for _ in
                                                        range(len(self.padding))])
        self.process5_up = SpectralConv2d(n_hidden * 16 // factor, n_hidden * 16 // factor,
                                                      *[max(1, min(modes, min(self.augmented_resolution) // 32))
                                                        for _ in
                                                        range(len(self.padding))])
        self.w1_up = ConvList[len(self.padding)](n_hidden, n_hidden, 1)
        self.w2_up = ConvList[len(self.padding)](n_hidden * 2 // factor, n_hidden * 2 // factor, 1)
        self.w3_up = ConvList[len(self.padding)](n_hidden * 4 // factor, n_hidden * 4 // factor, 1)
        self.w4_up = ConvList[len(self.padding)](n_hidden * 8 // factor, n_hidden * 8 // factor, 1)
        self.w5_up = ConvList[len(self.padding)](n_hidden * 16 // factor, n_hidden * 16 // factor, 1)
        # projectors
        self.fc1 = nn.Linear(n_hidden, n_hidden * 2)
        self.fc2 = nn.Linear(n_hidden * 2, out_dim)

    def forward(self, x, coords):
        b, c, h, w = x.shape
        x = torch.cat((x, coords.repeat(b, 1, 1, 1)), dim=1)
        x = self.preprocess(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = F.pad(x, [0, self.padding[1], 0, self.padding[0]])
        x1 = self.inc(x)
        x1 = F.gelu(self.process1_down(x1) + self.w1_down(x1))
        x2 = self.down1(x1)
        x2 = F.gelu(self.process2_down(x2) + self.w2_down(x2))
        x3 = self.down2(x2)
        x3 = F.gelu(self.process3_down(x3) + self.w3_down(x3))
        x4 = self.down3(x3)
        x4 = F.gelu(self.process4_down(x4) + self.w4_down(x4))
        x5 = self.down4(x4)
        x5 = F.gelu(self.process5_down(x5) + self.w5_down(x5))
        x5 = F.gelu(self.process5_up(x5) + self.w5_up(x5))
        x = self.up1(x5, x4)
        x = F.gelu(self.process4_up(x) + self.w4_up(x))
        x = self.up2(x, x3)
        x = F.gelu(self.process3_up(x) + self.w3_up(x))
        x = self.up3(x, x2)
        x = F.gelu(self.process2_up(x) + self.w2_up(x))
        x = self.up4(x, x1)
        x = F.gelu(self.process1_up(x) + self.w1_up(x))
        x = self.outc(x)

        if self.padding[0] > 0:
            x = x[..., :-self.padding[0], :]
        if self.padding[1] > 0:
            x = x[..., :, :-self.padding[1]]

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2) 

