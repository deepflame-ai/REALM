"""
Reference:
    Transolver: A Fast Transformer Solver for PDEs on General Geometries (ICML'2024).
    Source: https://github.com/thuml/Neural-Solver-Library/blob/main/models/Transolver.py
"""
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from models.base._Basic import MLP
from models.base._Embedding import timestep_embedding, unified_pos_embedding


class Physics_Attention_Structured_Mesh_3D(nn.Module):
    ## for structured mesh in 3D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32, shapelist=None, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = shapelist[0]
        self.W = shapelist[1]
        self.D = shapelist[2]

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)
    

class Transolver_block(nn.Module):
    """Transolver encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout=0.0,
            act='gelu',
            mlp_ratio=2,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            shapelist=None
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.Attn = Physics_Attention_Structured_Mesh_3D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, slice_num=slice_num, shapelist=shapelist)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Transolver3d'

        self.in_dim = args.in_dim - 3 
        self.out_dim = args.out_dim
        self.hidden_dim = args.width
        self.num_layers = args.n_layers
        self.num_heads = args.n_heads
        self.shapelist = args.shape_list
        self.ref = 8


        self.pos = unified_pos_embedding(self.shapelist, self.ref)
        self.preprocess = MLP(self.in_dim + self.ref ** len(self.shapelist), self.hidden_dim * 2,
                                self.hidden_dim, n_layers=0, res=False)

        ## models
        self.blocks = nn.ModuleList([Transolver_block(num_heads=self.num_heads, hidden_dim=self.hidden_dim, 
                                                      out_dim=self.out_dim,
                                                      last_layer=(_ == self.num_layers - 1),
                                                      shapelist=self.shapelist) for _ in range(self.num_layers)])
    
        self.placeholder = nn.Parameter((1 / (self.hidden_dim)) * torch.rand(self.hidden_dim, dtype=torch.float))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, coords):
        b, c, h, w, l = x.shape
        coords = unified_pos_embedding(self.shapelist, self.ref)
        coords = coords.reshape(1, self.shapelist[0], self.shapelist[1], self.shapelist[2], -1).permute(0, 4, 1, 2, 3)  # [1, H, W, L, C]
        x = torch.cat([x, coords.repeat(b, 1, 1, 1, 1)], dim=1)      # [B, C+2, H, W, L]
        x = x.permute(0, 2, 3, 4, 1).reshape(b, h * w * l, -1)            # [B, H*W*L, C+2]
        x = self.preprocess(x)
        x = x + self.placeholder[None, None, :]
        for block in self.blocks:
            x = block(x)
        x = x.view(b, h, w, l, -1).permute(0, 4, 1, 2, 3)            # [B, out_dim, H, W]
        return x