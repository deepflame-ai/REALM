import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from models.base._Basic import MLP, LinearAttention, FlashAttention, SelfAttention as LinearSelfAttention
from models.base._Embedding import timestep_embedding, unified_pos_embedding
import warnings


def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.linalg.cholesky(A, upper=upper, out=out)
        if torch.isnan(L).any():
            raise RuntimeError
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise ValueError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(10):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        raise e


class ONOBlock(nn.Module):
    """ONO encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout=0.0,
            act='gelu',
            attn_type='selfAttention',
            mlp_ratio=2,
            last_layer=False,
            momentum=0.9,
            psi_dim=8,
            out_dim=1
    ):
        super().__init__()
        self.momentum = momentum
        self.psi_dim = psi_dim

        self.register_buffer("feature_cov", torch.zeros(psi_dim, psi_dim))
        self.register_buffer("init_count", torch.zeros(1, dtype=torch.long))
        self.register_parameter("mu", nn.Parameter(torch.zeros(psi_dim)))
        self.ln_1 = nn.LayerNorm(hidden_dim)
        if attn_type == 'nystrom':
            from nystrom_attention import NystromAttention
            self.Attn = NystromAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout)
        elif attn_type == 'linear':
            self.Attn = LinearAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout,
                                        attn_type='galerkin')
        elif attn_type == 'selfAttention':
            self.Attn = LinearSelfAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout)
        else:
            raise ValueError('Attn type only supports nystrom or linear')
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        self.proj = nn.Linear(hidden_dim, psi_dim)
        self.ln_3 = nn.LayerNorm(hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, out_dim) if last_layer else MLP(hidden_dim, hidden_dim * mlp_ratio,
                                                                          hidden_dim, n_layers=0, res=False, act=act)

    def forward(self, x, fx):
        x = self.Attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        x_ = self.proj(x)
        if self.training:
            batch_cov = torch.einsum("blc, bld->cd", x_, x_) / x_.shape[0] / x_.shape[1]
            with torch.no_grad():
                if self.init_count.item() == 0:
                    self.feature_cov = batch_cov
                    self.init_count.add_(1)
                else:
                    self.feature_cov.mul_(self.momentum).add_(batch_cov, alpha=1 - self.momentum)
        else:
            batch_cov = self.feature_cov
        L = psd_safe_cholesky(batch_cov)
        L_inv_T = L.inverse().transpose(-2, -1)
        x_ = x_ @ L_inv_T

        fx = (x_ * torch.nn.functional.softplus(self.mu)) @ (x_.transpose(-2, -1) @ fx) + fx
        fx = self.mlp2(self.ln_3(fx))

        return x, fx


class Model(nn.Module):
    ## speed up with flash attention
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'ONO'
        self.in_dim = args.in_dim - 2 
        self.out_dim = args.out_dim
        self.hidden_dim = args.width
        self.num_layers = args.n_layers
        self.num_heads = args.n_heads
        self.shapelist = args.shape_list
        self.ref = 8


        self.pos = unified_pos_embedding(self.shapelist, self.ref)
        self.preprocess_x = MLP(self.in_dim + self.ref ** len(self.shapelist), self.hidden_dim * 2,
                                self.hidden_dim, n_layers=0, res=False)

        self.preprocess_z = MLP(self.in_dim + self.ref ** len(self.shapelist), self.hidden_dim * 2,
                                self.hidden_dim, n_layers=0, res=False)

        ## models
        self.blocks = nn.ModuleList([ONOBlock(num_heads=self.num_heads, hidden_dim=self.hidden_dim,
                                              out_dim=self.out_dim,
                                              last_layer=(_ == self.num_layers - 1))
                                     for _ in range(self.num_layers)])
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
        b, c, h, w = x.shape
        coords = unified_pos_embedding(self.shapelist, self.ref)
        coords = coords.reshape(1, self.shapelist[0], self.shapelist[1], -1).repeat(b, 1, 1, 1)  # [1, H, W, C]
        x = torch.cat([coords.permute(0, 3, 1, 2), x], dim=1)      # [B, C+2, H, W]
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, -1)       

        fx = self.preprocess_z(x)
        x = self.preprocess_x(x)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            x, fx = block(x, fx)
        fx = fx.view(b, h, w, -1).permute(0, 3, 1, 2)            # [B, out_dim, H, W]
        return fx