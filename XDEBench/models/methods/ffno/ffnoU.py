import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from models.base._Basic import MLP
from models.base._Embedding import timestep_embedding, unified_pos_embedding
from models.base._FFNO_Layers import SpectralConv1d, SpectralConv2d, SpectralConv3d
from models.base._GeoFNO_Projection import SpectralConv2d_IrregularGeo, IPHI

BlockList = [None, SpectralConv1d, SpectralConv2d, SpectralConv3d]
ConvList = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d]


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'FFNOU'
        self.args = args
        s1=96
        s2=96
        
        ## embedding
        self.preprocess = MLP(args.in_dim, args.width * 2, args.width,
                              n_layers=0, res=False, act=args.act)
        
        # geometry projection
        self.fftproject_in = SpectralConv2d_IrregularGeo(args.width, args.width, args.modes, args.modes, s1, s2)
        self.fftproject_out = SpectralConv2d_IrregularGeo(args.width, args.width, args.modes, args.modes, s1, s2)
        self.iphi = IPHI()
        self.padding = [(16 - size % 16) % 16 for size in [s1, s2]]

        self.spectral_layers = nn.ModuleList([])
        for _ in range(args.n_layers):
            self.spectral_layers.append(BlockList[len(self.padding)](args.width, args.width,
                                                                     *[args.modes for _ in range(len(self.padding))]))
        # projectors
        self.fc1 = nn.Linear(args.width, args.width)
        self.fc2 = nn.Linear(args.width, args.out_dim)

    def forward(self, x, coords, time=None):
        original_pos = coords.permute(0, 2, 1) # [1, num_points, dim]
        
        b, c, g = x.shape
        x = torch.cat([x, coords.repeat(b, 1, 1)], dim=1)
        x = x.permute(0, 2, 1).reshape(b, g, -1)
        x = self.preprocess(x) # [b, g, width]

        x = self.fftproject_in(x.permute(0, 2, 1), x_in=original_pos, iphi=self.iphi, code=None)
        for i in range(self.args.n_layers):
            x = x + self.spectral_layers[i](x)
        x = self.fftproject_out(x, x_out=original_pos, iphi=self.iphi, code=None).permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        x = x.view(b, g, -1).permute(0, 2, 1)
        return x