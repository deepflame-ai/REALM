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


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[2, 3])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        return x


class Model(nn.Module):  
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'FNO'
        self.modes1 = [args.modes] * 2
        self.modes2 = [args.modes] * 2
        self.layers = [args.width] * 3
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.num_chemical = args.num_chemical
        self.act = _get_act(args.act)

        self.fc0 = nn.Linear(self.in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([
            SpectralConv2d(in_size, out_size, m1, m2)
            for in_size, out_size, m1, m2 in zip(self.layers, self.layers[1:], self.modes1, self.modes2)
        ])
        self.ws = nn.ModuleList([
            nn.Conv1d(in_size, out_size, 1)
            for in_size, out_size in zip(self.layers, self.layers[1:])
        ])

        self.fc1 = nn.Linear(self.layers[-1], 128)
        self.fc2 = nn.Linear(128, self.out_dim)

    def forward(self, x, coords):
        size_1, size_2 = x.shape[-2], x.shape[-1]
        batchsize = x.shape[0]
        x = torch.cat((x, coords.repeat(batchsize, 1, 1, 1)), dim=1)  # concat coord
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_1, size_2)
            x = x1 + x2
            if i != len(self.ws) - 1:
                x = self.act(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)
