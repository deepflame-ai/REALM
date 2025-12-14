import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from models.base._Basic import MLP
from models.base._Embedding import timestep_embedding, unified_pos_embedding
from models.base._Neural_Spectral_Block import NeuralSpectralBlock1D, NeuralSpectralBlock2D, NeuralSpectralBlock3D
from models.base._UNet_Blocks import DoubleConv1D, Down1D, Up1D, OutConv1D, DoubleConv2D, Down2D, Up2D, OutConv2D, \
    DoubleConv3D, Down3D, Up3D, OutConv3D
from models.base._GeoFNO_Projection import SpectralConv2d_IrregularGeo, IPHI

ConvList = [None, DoubleConv1D, DoubleConv2D, DoubleConv3D]
DownList = [None, Down1D, Down2D, Down3D]
UpList = [None, Up1D, Up2D, Up3D]
OutList = [None, OutConv1D, OutConv2D, OutConv3D]
BlockList = [None, NeuralSpectralBlock1D, NeuralSpectralBlock2D, NeuralSpectralBlock3D]


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'LSM'
        self.args = args
        bilinear=True
        num_token=4
        num_basis=12
        s1=96
        s2=96
        ref=8
        time_input=False
        normtype = 'in' # when conducting dynamic tasks, use instance norm for stability
            
        ## embedding
        self.preprocess = MLP(args.in_dim, args.width * 2, args.width, n_layers=0, res=False, act=args.act)
        
        if time_input:
            self.time_fc = nn.Sequential(nn.Linear(args.width, args.width), nn.SiLU(),
                                         nn.Linear(args.width, args.width))
            
        # geometry projection
        self.fftproject_in = SpectralConv2d_IrregularGeo(args.width, args.width, args.modes, args.modes,
                                                            s1, s2)
        self.fftproject_out = SpectralConv2d_IrregularGeo(args.width, args.width, args.modes, args.modes,
                                                            s1, s2)
        self.iphi = IPHI()
        patch_size = [(size + (16 - size % 16) % 16) // 16 for size in [s1, s2]]
        self.padding = [(16 - size % 16) % 16 for size in [s1, s2]]

        # multiscale modules
        self.inc = ConvList[len(patch_size)](args.width, args.width, normtype=normtype)
        self.down1 = DownList[len(patch_size)](args.width, args.width * 2, normtype=normtype)
        self.down2 = DownList[len(patch_size)](args.width * 2, args.width * 4, normtype=normtype)
        self.down3 = DownList[len(patch_size)](args.width * 4, args.width * 8, normtype=normtype)
        factor = 2 if bilinear else 1
        self.down4 = DownList[len(patch_size)](args.width * 8, args.width * 16 // factor, normtype=normtype)
        self.up1 = UpList[len(patch_size)](args.width * 16, args.width * 8 // factor, bilinear, normtype=normtype)
        self.up2 = UpList[len(patch_size)](args.width * 8, args.width * 4 // factor, bilinear, normtype=normtype)
        self.up3 = UpList[len(patch_size)](args.width * 4, args.width * 2 // factor, bilinear, normtype=normtype)
        self.up4 = UpList[len(patch_size)](args.width * 2, args.width, bilinear, normtype=normtype)
        self.outc = OutList[len(patch_size)](args.width, args.width)
        # Patchified Neural Spectral Blocks
        self.process1 = BlockList[len(patch_size)](args.width, num_basis, patch_size, num_token, args.n_heads)
        self.process2 = BlockList[len(patch_size)](args.width * 2, num_basis, patch_size, num_token, args.n_heads)
        self.process3 = BlockList[len(patch_size)](args.width * 4, num_basis, patch_size, num_token, args.n_heads)
        self.process4 = BlockList[len(patch_size)](args.width * 8, num_basis, patch_size, num_token, args.n_heads)
        self.process5 = BlockList[len(patch_size)](args.width * 16 // factor, num_basis, patch_size, num_token,
                                                   args.n_heads)
        # projectors
        self.fc1 = nn.Linear(args.width, args.width * 2)
        self.fc2 = nn.Linear(args.width * 2, args.out_dim)

    def forward(self, x, coords, time=None):
        original_pos = coords.permute(0, 2, 1)
        
        b, c, g = x.shape
        x = torch.cat([x, coords.repeat(b, 1, 1)], dim=1)
        x = x.permute(0, 2, 1).reshape(b, g, -1)
        x = self.preprocess(x)

        if time is not None:
            Time_emb = timestep_embedding(time, self.args.width).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            x = x + Time_emb

        x = self.fftproject_in(x.permute(0, 2, 1), x_in=original_pos, iphi=self.iphi, code=None)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        x = self.outc(x)
        x = self.fftproject_out(x, x_out=original_pos, iphi=self.iphi, code=None).permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.view(b, g, -1).permute(0, 2, 1)

        return x
