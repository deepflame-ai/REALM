import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, factor, n_layers, layer_norm):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes1, n_modes2, factor=4,
                 n_ff_layers=2, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes1 = n_modes1
        self.n_modes2 = n_modes2

        self.fourier_weight = nn.ParameterList([])

        weight1 = torch.FloatTensor(in_dim, out_dim, n_modes2, 2)
        param1 = nn.Parameter(weight1)
        nn.init.xavier_normal_(param1)
        self.fourier_weight.append(param1)

        weight2 = torch.FloatTensor(in_dim, out_dim, n_modes1, 2)
        param2 = nn.Parameter(weight2)
        nn.init.xavier_normal_(param2)
        self.fourier_weight.append(param2)

        self.backcast_ff = FeedForward(out_dim, factor, n_ff_layers, layer_norm)

    def forward(self, x):
        x = self.forward_fourier(x)
        b = self.backcast_ff(x)
        return b

    def forward_fourier(self, x):
        x = rearrange(x, 'b m n i -> b i m n')
        B, I, M, N = x.shape
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_fty.new_zeros(B, self.out_dim, M, N//2+1)

        out_ft[:, :, :, :self.n_modes2] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :self.n_modes2],
                torch.view_as_complex(self.fourier_weight[0]))

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_ftx.new_zeros(B, self.out_dim, M//2+1, N)
        out_ft[:, :, :self.n_modes1, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :self.n_modes1, :],
                torch.view_as_complex(self.fourier_weight[1]))

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        x = xx + xy
        x = rearrange(x, 'b i m n -> b m n i')
        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.__name__ = 'FFNO'
        if isinstance(args.modes, int):
            self.modes1 = self.modes2 = args.modes
        else:
            self.modes1, self.modes2 = args.modes
        self.width = args.width
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.num_chemical = args.num_chemical
        self.n_layers = args.n_layers

        self.in_proj = nn.Linear(self.in_dim, self.width)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(self.n_layers):
            self.spectral_layers.append(SpectralConv2d(in_dim=self.width,
                                                       out_dim=self.width,
                                                       n_modes1=self.modes1,
                                                       n_modes2=self.modes2))
        self.out = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, self.out_dim))
    
    def forward(self, x, coords):
        batch_size = x.shape[0]
        x = torch.cat((x, coords.repeat(batch_size, 1, 1, 1)), dim=1)
        x = self.in_proj(x.permute(0, 2, 3, 1))
        x = F.gelu(x)
        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b = layer(x)
            x = x + b    
        x = self.out(x).permute(0, 3, 1, 2)
        return x