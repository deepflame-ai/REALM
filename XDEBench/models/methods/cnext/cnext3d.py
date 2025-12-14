"""
Mixed adaptation from:

    Liu et al. 2022, A ConvNet for the 2020s.
    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Ronneberger et al., 2015. Convolutional Networks for Biomedical Image Segmentation.

If you use this implementation, please cite original work above.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from huggingface_hub import PyTorchModelHubMixin
from torch.utils.checkpoint import checkpoint


class BaseModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
    ):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.spatial_resolution = spatial_resolution
conv_modules = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

permute_channel_strings = {
    2: [
        "N C H W -> N H W C",
        "N H W C -> N C H W",
    ],
    3: [
        "N C D H W -> N D H W C",
        "N D H W C -> N C D H W",
    ],
}


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self, normalized_shape, n_spatial_dims, eps=1e-6, data_format="channels_last"
    ):
        super().__init__()
        if data_format == "channels_last":
            padded_shape = (normalized_shape,)
        else:
            padded_shape = (normalized_shape,) + (1,) * n_spatial_dims
        self.weight = nn.Parameter(torch.ones(padded_shape))
        self.bias = nn.Parameter(torch.zeros(padded_shape))
        self.n_spatial_dims = n_spatial_dims
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            x = F.normalize(x, p=2, dim=1, eps=self.eps) * self.weight
            return x


class Upsample(nn.Module):
    r"""Upsample layer."""

    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_transpose_modules[n_spatial_dims](
                dim_in, dim_out, kernel_size=2, stride=2
            ),
        )

    def forward(self, x):
        return self.block(x)


class Downsample(nn.Module):
    r"""Downsample layer."""

    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_modules[n_spatial_dims](dim_in, dim_out, kernel_size=2, stride=2),
        )
        self.n_spatial_dims = n_spatial_dims

    def forward(self, x):
        pad_needed = []
        for i in range(self.n_spatial_dims):
            dim_size = x.shape[2 + i]
            if dim_size % 2 != 0:
                pad_needed.append((0, 1))
            else:
                pad_needed.append((0, 0))
        
        if any(p[1] != 0 for p in pad_needed):
            pad = []
            for p in reversed(pad_needed):
                pad.extend(p)
            x = F.pad(x, pad, mode='constant', value=0)
        
        return self.block(x)


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, n_spatial_dims, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.dwconv = conv_modules[n_spatial_dims](
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, n_spatial_dims, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = rearrange(x, permute_channel_strings[self.n_spatial_dims][0])
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # (N, H, W, C) -> (N, C, H, W)
        x = rearrange(x, permute_channel_strings[self.n_spatial_dims][1])
        x = input + self.drop_path(x)
        return x


class Stage(nn.Module):
    r"""ConvNeXt Stage.
    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        n_spatial_dims (int): Number of spatial dimensions.
        depth (int): Number of blocks in the stage.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mode (str): Down, Up, Neck. Default: "down"
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        n_spatial_dims,
        depth=1,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        mode="down",
        skip_project=False,
    ):
        super().__init__()

        if skip_project:
            self.skip_proj = conv_modules[n_spatial_dims](2 * dim_in, dim_in, 1)
        else:
            self.skip_proj = nn.Identity()
        if mode == "down":
            self.resample = Downsample(dim_in, dim_out, n_spatial_dims)
        elif mode == "up":
            self.resample = Upsample(dim_in, dim_out, n_spatial_dims)
        else:
            self.resample = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                Block(dim_in, n_spatial_dims, drop_path, layer_scale_init_value)
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        x = self.skip_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.resample(x)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'CNext3d'
        blocks_per_stage = 1
        blocks_at_neck = 1
        n_spatial_dims=3
        dim_in = args.in_dim
        dim_out = args.out_dim
        features = args.width
        stages = args.n_layers

        encoder_dims = [features * 2**i for i in range(stages + 1)]
        decoder_dims = [features * 2**i for i in range(stages, -1, -1)]
        encoder = []
        decoder = []
        self.in_proj = conv_modules[n_spatial_dims](
            dim_in, features, kernel_size=3, padding=1
        )
        self.out_proj = conv_modules[n_spatial_dims](
            features, dim_out, kernel_size=3, padding=1
        )
        for i in range(stages):
            encoder.append(
                Stage(
                    encoder_dims[i],
                    encoder_dims[i + 1],
                    n_spatial_dims,
                    blocks_per_stage,
                    mode="down",
                )
            )
            decoder.append(
                Stage(
                    decoder_dims[i],
                    decoder_dims[i + 1],
                    n_spatial_dims,
                    blocks_per_stage,
                    mode="up",
                    skip_project=i != 0,
                )
            )
        self.encoder = nn.ModuleList(encoder)
        self.neck = Stage(
            encoder_dims[-1],
            encoder_dims[-1],
            n_spatial_dims,
            blocks_at_neck,
            mode="neck",
        )
        self.decoder = nn.ModuleList(decoder)

        self.stages = stages

    def forward(self, x, coords):
        original_shape = x.shape[2:]
        d, h, w = original_shape
        
        max_scale = 2**self.stages
        pad_d = (max_scale - d % max_scale) % max_scale
        pad_h = (max_scale - h % max_scale) % max_scale
        pad_w = (max_scale - w % max_scale) % max_scale
        
        pad = [0, pad_w, 0, pad_h, 0, pad_d]
        if any(pad):
            x = F.pad(x, pad, mode='constant', value=0)
            padded_coords = F.pad(coords, pad, mode='constant', value=0)
        else:
            padded_coords = coords
        
        b, c, _, _, _ = x.shape
        if padded_coords.shape[0] != b:
            padded_coords = padded_coords.repeat(b, 1, 1, 1, 1)
        
        x = torch.cat((x, padded_coords), dim=1)
        x = self.in_proj(x)
        skips = []
        for i, enc in enumerate(self.encoder):
            skips.append(x)
            x = enc(x)
        x = self.neck(x)
        for j, dec in enumerate(self.decoder):
            if j > 0:
                current_size = x.shape[2:]
                skip_size = skips[-j].shape[2:]

                diff_d = skip_size[0] - current_size[0]
                diff_h = skip_size[1] - current_size[1]
                diff_w = skip_size[2] - current_size[2]

                start_d = diff_d // 2
                end_d = start_d + current_size[0]
                start_h = diff_h // 2
                end_h = start_h + current_size[1]
                start_w = diff_w // 2
                end_w = start_w + current_size[2]
                
                skip_adjusted = skips[-j][:, :, start_d:end_d, start_h:end_h, start_w:end_w]
                x = torch.cat([x, skip_adjusted], dim=1)
            x = dec(x) 
        x = self.out_proj(x)
        if any(pad):
            slices = [slice(None), slice(None)]
            slices.append(slice(0, original_shape[0]))  # D
            slices.append(slice(0, original_shape[1]))  # H
            slices.append(slice(0, original_shape[2]))  # W
            x = x[slices]
        return x