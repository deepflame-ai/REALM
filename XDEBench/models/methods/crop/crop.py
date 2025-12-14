import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import os
import numpy as np

# Removed utilities3 as it's not provided and likely contains helper functions
# that might be integrated or replaced.
# from utilities3 import *
from timeit import default_timer


# Helper function for activation (from fno.py)
def _get_act(act):
    if act == 'tanh':
        return F.tanh
    elif act == 'gelu':
        return F.gelu
    elif act == 'relu':
        return F.relu_
    elif act == 'elu':
        return F.elu_
    elif act == 'leaky_relu':
        return F.leaky_relu_
    else:
        raise ValueError(f'{act} is not supported')


################################################################
# fourier layer
################################################################

class Spectral_weights(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        dtype = torch.cfloat
        self.kernel_size_Y = 2 * modes1 - 1
        self.kernel_size_X = modes2
        
        # Initializing parameters similar to fno.py's SpectralConv2d
        self.W = nn.ParameterDict({
            'y0_modes': nn.Parameter(torch.empty(in_channels, out_channels, modes1 - 1, 1, dtype=dtype)),
            'yposx_modes': nn.Parameter(torch.empty(in_channels, out_channels, self.kernel_size_Y, self.kernel_size_X - 1, dtype=dtype)),
            '00_modes': nn.Parameter(torch.empty(in_channels, out_channels, 1, 1, dtype=torch.float))
        })
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        # Using kaiming_uniform_ as in original crop file
        for v in self.W.values():
            nn.init.kaiming_uniform_(v, a=math.sqrt(5))
            
    def get_weight(self):
        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        # Building the full weight matrix as in original crop file
        self.weights = torch.cat([self.W["y0_modes"], self.W["00_modes"].cfloat(), self.W["y0_modes"].flip(dims=(-2, )).conj()], dim=-2)
        self.weights = torch.cat([self.weights, self.W["yposx_modes"]], dim=-1)
        self.weights = self.weights.view(self.in_channels, self.out_channels,
                                         self.kernel_size_Y, self.kernel_size_X)
        

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        
        # Using Spectral_weights class for managing weights as in original crop file
        self.spectral_weight = Spectral_weights(in_channels=in_channels, out_channels=out_channels, modes1=modes1, modes2=modes2)
        self.get_weight()

    def get_weight(self):
        self.spectral_weight.get_weight()
        self.weights = self.spectral_weight.weights
        
    # Complex multiplication (from fno.py)
    @torch.jit.script
    def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        # This part is specific to the original crop file's handling of frequencies
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero(as_tuple=True)[0].item() # Added as_tuple=True for newer PyTorch versions
        self.get_weight()
        
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[..., (freq0_y - self.modes1 + 1):(freq0_y + self.modes1), :self.modes2]
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)              
        out_ft[..., (freq0_y - self.modes1 + 1):(freq0_y + self.modes1), :self.modes2] = \
            self.compl_mul2d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x) # Using F.gelu directly as in original crop file
        x = self.mlp2(x)
        return x
        
class Crop_to_latent_size(nn.Module):
    def __init__(self, in_size, out_size):
        super(Crop_to_latent_size, self).__init__()
        # in_size, out_size: tuple (H, W)
        if isinstance(in_size, int):
            in_size = (in_size, in_size)
        if isinstance(out_size, int):
            out_size = (out_size, out_size)
        self.in_size = in_size
        self.out_size = out_size
        self.temp_size_h = min(in_size[0], out_size[0])
        self.temp_size_w = min(in_size[1], out_size[1])

    def forward(self, u1):
        B, C, H, W = u1.shape
        fu1 = torch.fft.rfft2(u1, norm="ortho")
        fu1_recover = torch.zeros(
            (B, C, self.out_size[0], self.out_size[1] // 2 + 1),
            dtype=fu1.dtype, device=u1.device
        )
        # 填充频域系数
        fu1_recover[:, :, :self.temp_size_h // 2, :self.temp_size_w // 2 + 1] = fu1[:, :, :self.temp_size_h // 2, :self.temp_size_w // 2 + 1]
        fu1_recover[:, :, -self.temp_size_h // 2:, :self.temp_size_w // 2 + 1] = fu1[:, :, -self.temp_size_h // 2:, :self.temp_size_w // 2 + 1]
        # 逆变换和缩放
        u1_recover = torch.fft.irfft2(fu1_recover, s=self.out_size, norm="ortho") * (
            (self.out_size[0] * self.out_size[1]) / (self.in_size[0] * self.in_size[1])
        )
        return u1_recover

# Renamed FNO2d to Model for consistency with fno.py
class Model(nn.Module): 
    # Added args to init for consistency with fno.py. The original crop file 
    # passed modes1, modes2, width, in_size, latent_size directly.
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'CROP' # Renamed for clarity
        
        # Mapped args to original FNO2d parameters
        self.modes1 = args.modes1 if hasattr(args, "modes1") else args.modes
        self.modes2 = args.modes2 if hasattr(args, "modes2") else args.modes
        self.width = args.width
        self.out_dim = args.out_dim
        self.in_dim = args.in_dim

        self.in_size = (args.shape_list[0], args.shape_list[1])
        self.latent_size = (128, 128) 
        self.num_chemical = args.num_chemical
        self.act = _get_act(args.act) # Using _get_act for activation
        
        self.CROP_to_latent = Crop_to_latent_size(self.in_size, self.latent_size)
        self.CROP_back = Crop_to_latent_size(self.latent_size, self.in_size)
        
        # Original fc0 is now p
        self.p = nn.Conv2d(self.in_dim, self.width, 1) # args.in_channels should be 10 for original input

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.norm = nn.InstanceNorm2d(self.width)
        
        # Original fc1 and fc2 combined into q (MLP)
        self.q = MLP(self.width, self.out_dim, self.width * 4) # args.out_channels should be 1 for original output

    def forward(self, x, coords):
        batchsize = x.shape[0]
        x = torch.cat((x, coords.repeat(batchsize, 1, 1, 1)), dim=1)  # concat coord        
        x = self.CROP_to_latent(x)
        x = self.p(x)

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.act(x) # Using self.act for activation

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = self.CROP_back(x)
        x = self.q(x)
        return x