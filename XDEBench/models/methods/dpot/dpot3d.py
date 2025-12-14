"""
Reference:
    DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training (ICML'2024).
    Source: https://github.com/thu-ml/DPOT/blob/main/models/dpot3d.py
"""

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

import math
import logging

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU(),'silu':nn.SiLU()}


class AFNO3D(nn.Module):
    def __init__(self, width = 32, num_blocks=8, channel_first = False,sparsity_threshold=0.01, modes = 32, temporal_modes = 8,hard_thresholding_fraction=1, hidden_size_factor=1, act='gelu'):
        super(AFNO3D, self).__init__()
        assert width % num_blocks == 0, f"hidden_size {width} should be divisble by num_blocks {num_blocks}"


        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.temporal_modes = temporal_modes
        self.hidden_size_factor = hidden_size_factor
        self.act = ACTIVATION[act]
        # self.scale = 0.02
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        self.w1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size))

    ### N, C, X, Y, Z
    def forward(self, x, spatial_size=None):
        if self.channel_first:
            x = rearrange(x, 'b c x y z -> b x y z c')
        B, H, W, L, C = x.shape
        x_orig = x

        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")

        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # total_modes = H*W // 2 + 1
        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes,:self.temporal_modes] = F.gelu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes,:self.temporal_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes,:self.temporal_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes,:self.temporal_modes] = F.gelu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes,:self.temporal_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes,:self.temporal_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes, :kept_modes,:self.temporal_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes,:self.temporal_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes,:self.temporal_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes,:self.temporal_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes,:self.temporal_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes,:self.temporal_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        # x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], C)
        x = torch.fft.irfftn(x, s=(H, W, L), dim=(1, 2, 3), norm="ortho")

        x = x + x_orig
        if self.channel_first:
            x = rearrange(x, 'b x y t c -> b c x y t')
        return x




_logger = logging.getLogger(__name__)




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x





class TimeAggregator(nn.Module):
    def __init__(self, n_channels, n_timesteps, out_channels, type='mlp'):
        super(TimeAggregator, self).__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.type = type
        if self.type == 'mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
        elif self.type == 'exp_mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
            self.gamma = nn.Parameter(2**torch.linspace(-10,10, out_channels).unsqueeze(0),requires_grad=True)  # 1, C


    ##  B, X, Y, T, C
    def forward(self, x):
        if self.type == 'mlp':
            x = torch.einsum('tij, ...ti->...j', self.w, x)
        elif self.type == 'exp_mlp':
            t = torch.linspace(0, 1, x.shape[-2]).unsqueeze(-1).to(x.device) # T, 1
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum('tij,...ti->...j', self.w, x * t_embed)

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=128, act='gelu'):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * ( img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.out_dim = out_dim
        self.act = ACTIVATION[act]

        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv3d(embed_dim, out_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        B, C, H, W, L = x.shape
        assert H == self.img_size[0] and W == self.img_size[1] and L == self.img_size[2], f"Input image size ({H}*{W}*{L}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, mixing_type = 'afno', double_skip = True, width = 32, n_blocks = 4, mlp_ratio=1., channel_first = True, modes = 32, drop=0., drop_path=0., act='gelu', h=14, w=8,):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(8, width)
        self.width = width
        self.modes = modes
        self.act = ACTIVATION[act]

        if mixing_type == "afno":
            self.filter = AFNO3D(width = width, num_blocks=n_blocks, sparsity_threshold=0.01, channel_first = channel_first, modes = modes,
                                 hard_thresholding_fraction=1, hidden_size_factor=1, act=act)


        self.norm2 = torch.nn.GroupNorm(8, width)



        mlp_hidden_dim = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv3d(in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1),
        )

        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)


        if self.double_skip:
            x = x + residual
            residual = x

        # x = self.mlp(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.norm2(x)
        x = self.mlp(x)

        # x = self.drop_path(x)
        x = x + residual

        return x



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'DPOT3d'
        self.args = args


        self.in_channels = args.in_dim - 3
        self.out_channels = args.out_dim
        self.in_timesteps = 1
        self.out_timesteps = 1
        self.n_blocks = 8
        self.depth = args.n_layers
        self.num_features = self.embed_dim = args.width
        self.mlp_ratio = 2

        img_size = args.shape_list
        patch_size = 8
        self.act = ACTIVATION[args.act]
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=self.in_channels + 4,
            embed_dim=self.out_channels * patch_size + 4,
            out_dim=self.embed_dim, act=args.act
        )
        self.latent_size = self.patch_embed.out_size
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.embed_dim, self.patch_embed.out_size[0], self.patch_embed.out_size[1], self.patch_embed.out_size[2]
        ))

        self.normalize = False
        self.time_agg = 'mlp'
        self.mixing_type = 'afno'
        self.n_cls = 12
        self.out_layer_dim=64


        h = img_size[0] // patch_size
        w = img_size[1] // patch_size

        if isinstance(args.modes, int):
            self.modes = args.modes
        else:
            self.modes = args.modes[0]

        self.blocks = nn.ModuleList([
            Block(mixing_type=self.mixing_type,modes=self.modes,
                width=self.embed_dim, mlp_ratio=self.mlp_ratio, channel_first = True, n_blocks=self.n_blocks,
                double_skip=False, h=h, w=w,act = args.act)
            for i in range(self.depth)])


        if self.normalize:
            self.scale_feats_mu = nn.Linear(2 * self.in_channels, self.embed_dim)
            self.scale_feats_sigma = nn.Linear(2 * self.in_channels, self.embed_dim)


        self.cls_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            self.act,
            nn.Linear(self.embed_dim, self.embed_dim),
            self.act,
            nn.Linear(self.embed_dim, self.n_cls)
        )

        self.time_agg_layer = TimeAggregator(self.in_channels, self.in_timesteps, self.embed_dim, self.time_agg)


        ### attempt load balancing for high resolution
        self.out_layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.embed_dim, out_channels=self.out_layer_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv3d(in_channels=self.out_layer_dim, out_channels=self.out_layer_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv3d(in_channels=self.out_layer_dim, out_channels=self.out_channels * self.out_timesteps,kernel_size=1, stride=1)
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=.002)    # .02
            if m.bias is not None:
            # if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid


    def get_grid_4d(self, x):
        batchsize, size_x, size_y, size_z, size_t = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1, 1).to(x.device).repeat([batchsize, 1, size_y, size_z, size_t, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1, 1).to(x.device).repeat([batchsize, size_x, 1, size_z, size_t, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1, 1).to(x.device).repeat([batchsize, size_x, size_y, 1, size_t, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, 1, 1, 1, size_t, 1).to(x.device).repeat([batchsize, size_x, size_y, size_z, 1, 1])

        grid = torch.cat((gridx, gridy, gridz, gridt), dim=-1)
        return grid


    ### in/out: B, X, Y, T, C
    def forward(self, x, coords):
        x = x.permute(0, 2, 3, 4, 1).unsqueeze(4)
        B, _, _, _, T, _ = x.shape
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3,4),keepdim=True), x.std(dim=(1,2,3,4),keepdim=True) + 1e-6    # B,1,1,1,1,C
            x = (x - mu)/ sigma
            scale_mu = self.scale_feats_mu(torch.cat([mu, sigma],dim=-1)).squeeze(-2).permute(0,4,1,2,3)   #-> B, C, 1, 1, 1
            scale_sigma = self.scale_feats_sigma(torch.cat([mu, sigma], dim=-1)).squeeze(-2).permute(0, 4, 1, 2, 3)

        grid = self.get_grid_4d(x)
        x = torch.cat((x, grid), dim=-1).contiguous() # B, X, Y, Z, T, C+4
        x = rearrange(x, 'b x y z t c -> (b t) c x y z')
        x = self.patch_embed(x)

        x = x + self.pos_embed

        x = rearrange(x, '(b t) c x y z -> b x y z t c', b=B, t=T)

        x = self.time_agg_layer(x)

        # x = self.pos_drop(x)
        x = rearrange(x, 'b x y z c -> b c x y z')

        if self.normalize:
            x = scale_sigma * x + scale_mu   ### Ada_in layer

        for blk in self.blocks:
            x = blk(x)

        x = self.out_layer(x).permute(0, 2, 3, 4, 1)
        x = x.reshape(*x.shape[:4], self.out_timesteps, self.out_channels).contiguous()

        if self.normalize:
            x = x * sigma  + mu

        x = x[:, :, :, :, 0, :].permute(0, 4, 1, 2, 3)

        return x

    def extra_repr(self) -> str:

        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr = string_repr + '(' + name + '): ' \
                              + 'tensor(' + str(tuple(p[1].shape)) + ', requires_grad=' + str(
                    p[1].requires_grad) + ')\n'

        return string_repr



def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb



