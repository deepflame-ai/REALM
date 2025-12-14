# fno.py

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bixy,ioxy->boxy", a, b)


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


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class Model(nn.Module):  
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'FNO3d'
        self.modes1 = [args.modes[0]] * 2
        self.modes2 = [args.modes[1]] * 2
        self.modes3 = [args.modes[2]] * 2
        self.layers = [args.width] * 3
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.num_chemical = args.num_chemical
        self.act = _get_act(args.act)

        self.fc0 = nn.Linear(self.in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([
            SpectralConv3d(in_size, out_size, m1, m2, m3)
            for in_size, out_size, m1, m2, m3 in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)
        ])
        self.ws = nn.ModuleList([
            nn.Conv1d(in_size, out_size, 1)
            for in_size, out_size in zip(self.layers, self.layers[1:])
        ])

        self.fc1 = nn.Linear(self.layers[-1], 128)
        self.fc2 = nn.Linear(128, self.out_dim)

    def forward(self, x, coords):
        size_1, size_2, size_3 = x.shape[-3], x.shape[-2], x.shape[-1]
        batchsize = x.shape[0]
        if coords.shape[0] != batchsize:
            x = torch.cat((x, coords.repeat(batchsize, 1, 1, 1, 1)), dim=1)  # concat coord
        else:
            x = torch.cat((x, coords), dim=1)
        x = self.fc0(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_1, size_2, size_3)
            x = x1 + x2
            if i != len(self.ws) - 1:
                x = self.act(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.permute(0, 4, 1, 2, 3)
