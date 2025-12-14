import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_act(act):
    if act == 'tanh':
        return torch.tanh
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


class Crop_to_latent_size_3d(nn.Module):
    """Frequency domain crop/pad for 3D tensors, supports list in_size/out_size."""
    def __init__(self, in_size, out_size):
        super().__init__()
        # 确保是 list
        self.in_size = list(in_size) if isinstance(in_size, (list, tuple)) else [in_size]*3
        self.out_size = list(out_size) if isinstance(out_size, (list, tuple)) else [out_size]*3
        self.temp_size = [min(i, o) for i, o in zip(self.in_size, self.out_size)]

    def forward(self, u1):
        # u1: (B, C, D, H, W)
        B, C, D, H, W = u1.shape
        fu1 = torch.fft.rfftn(u1, dim=(-3, -2, -1), norm="ortho")
        fu1_recover = torch.zeros(
            (B, C, self.out_size[0], self.out_size[1], self.out_size[2] // 2 + 1),
            dtype=torch.complex64, device=u1.device
        )

        td, th, tw = [t // 2 for t in self.temp_size]

        # 低频拷贝
        fu1_recover[:, :, :td, :th, :tw+1] = fu1[:, :, :td, :th, :tw+1]
        fu1_recover[:, :, -td:, :th, :tw+1] = fu1[:, :, -td:, :th, :tw+1]
        fu1_recover[:, :, :td, -th:, :tw+1] = fu1[:, :, :td, -th:, :tw+1]
        fu1_recover[:, :, -td:, -th:, :tw+1] = fu1[:, :, -td:, -th:, :tw+1]

        u1_recover = torch.fft.irfftn(
            fu1_recover,
            s=(self.out_size[0], self.out_size[1], self.out_size[2]),
            norm="ortho"
        ) * (
            (self.out_size[0] / self.in_size[0]) *
            (self.out_size[1] / self.in_size[1]) *
            (self.out_size[2] / self.in_size[2])
        )**(1/3)
        return u1_recover


class SpectralConv3d_CROP(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2, self.modes3 = modes[0], modes[1], modes[2]

        self.scale = (1 / (in_channels * out_channels))
        shape = (in_channels, out_channels, self.modes1, self.modes2, self.modes3)
        self.weights1 = nn.Parameter(self.scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(*shape, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(batchsize, self.out_channels,
                             x.size(-3), x.size(-2), x.size(-1)//2 + 1,
                             dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.__name__ = 'CROP3d'
        self.modes = args.modes
        self.width = args.width
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        # if args.random_crop:
        #     self.in_size = (args.crop_size, args.crop_size, args.crop_size)
        # else:
        #     self.in_size = args.shape_list
        self.in_size = args.shape_list
        self.latent_size = [int(args.shape_list[0]/2), 
                            int(args.shape_list[1]/2), 
                            int(args.shape_list[2]/2)]
        self.num_chemical = args.num_chemical
        self.act = _get_act(args.act)

        self.CROP_to_latent = Crop_to_latent_size_3d(self.in_size, self.latent_size)
        self.CROP_back = Crop_to_latent_size_3d(self.latent_size, self.in_size)

        self.p = nn.Conv3d(self.in_dim, self.width, 1)

        self.conv0 = SpectralConv3d_CROP(self.width, self.width, self.modes)
        self.conv1 = SpectralConv3d_CROP(self.width, self.width, self.modes)
        self.conv2 = SpectralConv3d_CROP(self.width, self.width, self.modes)
        self.conv3 = SpectralConv3d_CROP(self.width, self.width, self.modes)

        self.mlp0 = MLP3D(self.width, self.width, self.width)
        self.mlp1 = MLP3D(self.width, self.width, self.width)
        self.mlp2 = MLP3D(self.width, self.width, self.width)
        self.mlp3 = MLP3D(self.width, self.width, self.width)

        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.norm = nn.InstanceNorm3d(self.width)
        self.q = MLP3D(self.width, self.out_dim, self.width * 4)

    def forward(self, x, coords):
        batchsize = x.shape[0]
        if coords.shape[0] != batchsize:
            x = torch.cat((x, coords.repeat(batchsize, 1, 1, 1, 1)), dim=1)
        else:
            x = torch.cat((x, coords), dim=1)
        x = self.CROP_to_latent(x)
        x = self.p(x)

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = self.act(x1 + x2)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = self.act(x1 + x2)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = self.act(x1 + x2)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = self.CROP_back(x)
        x = self.q(x)
        return x
