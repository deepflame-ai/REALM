import glob
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import argparse
from tqdm import tqdm
import time
import os
import gc
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset, DataLoader, TensorDataset
# import logging, pickle, h5py

from models.base._Basic_FactFormer import FABlock2D



class GaussianFourierFeatureTransform(nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels,
                 mapping_size=256, scale=10, learnable=False,
                 num_heads=1):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size

        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size * num_heads)) * scale,
                               requires_grad=learnable)
        self.num_heads = num_heads

    def forward(self, x, unfold_head=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        if unfold_head:
            x = rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class FactorizedTransformer(nn.Module):
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 dim_out,
                 depth,
                 **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):

            layer = nn.ModuleList([])
            layer.append(nn.Sequential(
                GaussianFourierFeatureTransform(2, dim // 2, 1),
                nn.Linear(dim, dim)
            ))
            layer.append(FABlock2D(dim, dim_head, dim, heads, dim_out, use_rope=True,
                                   **kwargs))
            self.layers.append(layer)

    def forward(self, u, pos_lst):
        b, nx, ny, c = u.shape
        nx, ny = pos_lst[0].shape[0], pos_lst[1].shape[0]
        pos = torch.stack(torch.meshgrid([pos_lst[0].squeeze(-1), pos_lst[1].squeeze(-1)]), dim=-1)
        for pos_enc, attn_layer in self.layers:
            u += pos_enc(pos).view(1, nx, ny, -1)
            u = attn_layer(u, pos_lst).view(b, nx, ny, -1) + u
        return u


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'FactFormerv2'
        self.args = args

        self.in_dim = args.in_dim - 2 
        self.out_dim = args.out_dim

        self.resolution_x = args.shape_list[0]
        self.resolution_y = args.shape_list[1]

        self.depth = args.n_layers
        self.dim_head = args.width    
        self.heads = args.n_heads
        self.dim = args.width         


        self.pos_in_dim = 2
        self.pos_out_dim = 2
        self.positional_embedding = 'rotary'
        self.kernel_multiplier = 2

        self.to_in = nn.Linear(self.in_dim, self.dim, bias=True)

        self.encoder = FactorizedTransformer(self.dim, self.dim_head, self.heads, self.dim, self.depth,
                                             kernel_multiplier=self.kernel_multiplier)

        self.down_block = nn.Sequential(
            nn.InstanceNorm2d(self.dim),
            nn.Conv2d(self.dim, self.dim//2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.dim//2, self.dim//2, kernel_size=3, stride=1, padding=1, bias=True))

        self.up_block = nn.Sequential(
            nn.Upsample(size=(self.resolution_x, self.resolution_y), mode='nearest'),
            nn.Conv2d(self.dim//2, self.dim//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.dim//2, self.dim, kernel_size=3, stride=1, padding=1, bias=True))

        self.simple_to_out = nn.Sequential(
            Rearrange('b nx ny c -> b c (nx ny)'),
            nn.GroupNorm(num_groups=8, num_channels=self.dim*2),
            nn.Conv1d(self.dim*2, self.dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv1d(self.dim, self.out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self,
                u, corrds
                
                ):
        pos_x = 2 * torch.pi * torch.linspace(0, 1, self.resolution_x).float().unsqueeze(-1).to(u.device)
        pos_y = 2 * torch.pi * torch.linspace(0, 1, self.resolution_y).float().unsqueeze(-1).to(u.device)

        pos_lst = [pos_x, pos_y]
        u = rearrange(u, 'b c nx ny -> b nx ny c')
        b, nx, ny, c = u.shape
        u = self.to_in(u)
        u_last = self.encoder(u, pos_lst)
        u = rearrange(u_last, 'b nx ny c -> b c nx ny')
        u = self.down_block(u)
        u = self.up_block(u)
        u = rearrange(u, 'b c nx ny -> b nx ny c')
        u = torch.cat([u, u_last], dim=-1)
        u = self.simple_to_out(u)
        u = rearrange(u, 'b c (nx ny) -> b c nx ny', nx=nx, ny=ny)
        return u
