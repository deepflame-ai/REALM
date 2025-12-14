import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base._Basic import LinearAttention


def _get_act(act):
    if act == 'tanh':
        return nn.Tanh()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'elu':
        return nn.ELU(inplace=True)
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    else:
        raise ValueError(f'{act} is not supported')


class GNOTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, act, dropout=0.0, mlp_ratio=2):
        super(GNOTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.attn_type = 'linear'
        if self.attn_type == 'linear':
            self.attn = LinearAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout,
                                        attn_type='galerkin')
        else:
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            _get_act(act),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x):
        x_ = self.norm1(x)
        if self.attn_type == 'linear':
            x = self.attn(x_) + x
        else:
            x = x + self.attn(x_, x_, x_)[0]
        x = x + self.mlp(self.norm2(x))
        return self.norm3(x)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'GNOT'

        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.hidden_dim = args.width
        self.num_layers = args.n_layers
        self.num_heads = args.n_heads
        self.act = args.act
        self.coord_dim = args.coord_dim if hasattr(args, 'coord_dim') else 2

        self.embed = nn.Linear(self.in_dim, self.hidden_dim)
        self.gnot_blocks = nn.ModuleList([
            GNOTBlock(self.hidden_dim, self.num_heads, self.act)
            for _ in range(self.num_layers)
        ])
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            _get_act(self.act),
            nn.Linear(2 * self.hidden_dim, self.out_dim)
        )

    def forward(self, x, coords):
        b, c, h, w = x.shape
        x = torch.cat([x, coords.repeat(b, 1, 1, 1)], dim=1)  # [B, C+2, H, W]
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, -1)       # [B, H*W, C+2]

        if self.embed is None:
            in_features = x.shape[-1]
            self.embed = nn.Linear(in_features, self.hidden_dim).to(x.device)

        x = self.embed(x)
        for blk in self.gnot_blocks:
            x = blk(x)
        x = self.proj(x)
        x = x.view(b, h, w, -1).permute(0, 3, 1, 2)
        return x
