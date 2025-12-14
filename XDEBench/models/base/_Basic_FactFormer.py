import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class GeAct(nn.Module):
    """Gated activation function"""
    def __init__(self, act_fn):
        super().__init__()
        self.fn = act_fn

    def forward(self, x):
        c = x.shape[-1]  # channel last arrangement
        return self.fn(x[..., :int(c//2)]) * x[..., int(c//2):]


class MLP(nn.Module):
    def __init__(self, dims, act_fn, dropout=0.):
        super().__init__()
        layers = []

        for i in range(len(dims) - 1):
            if isinstance(act_fn, GeAct) and i < len(dims) - 2:
                layers.append(nn.Linear(dims[i], dims[i+1] * 2))
            else:
                layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(act_fn)
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def masked_instance_norm(x, mask, eps = 1e-6):
    """
    x of shape: [batch_size (N), num_objects (L), features(C)]
    mask of shape: [batch_size (N), num_objects (L), 1]
    """
    mask = mask.float()  # (N,L,1)
    mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))   # (N,C)
    mean = mean.detach()
    var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  #(N,C)
    var = var.detach()
    mean_reshaped = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
    var_reshaped = var.unsqueeze(1).expand_as(x)    # (N, L, C)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
    return ins_norm


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)


class GroupNorm(nn.Module):
    # group norm with channel at the last dimension
    def __init__(self, num_groups, num_channels,
                 domain_wise=False,
                 eps=1e-8, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.domain_wise = domain_wise
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(num_channels), requires_grad=True)

    def forward(self, x):
        # b h w c
        b, g_c = x.shape[0], x.shape[-1]
        c = g_c // self.num_groups
        if self.domain_wise:
            x = rearrange(x, 'b ... (g c) -> b g (... c)', g=self.num_groups)
        else:
            x = rearrange(x, 'b ... (g c) -> b ... g c', g=self.num_groups)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        if self.domain_wise:
            # b g (... c) -> b ... (g c)
            x = x.view(b, self.num_groups, -1, c)
            x = rearrange(x, 'b g ... c -> b ... (g c)')
        else:
            x = rearrange(x, 'b ... g c -> b ... (g c)',
                          g=self.num_groups)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class InstanceNorm(nn.Module):
    # instance norm with channel at the last dimension
    def __init__(self, num_channels, eps=1e-6, affine=False):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(num_channels), requires_grad=True)

    def forward(self, x):
        # b h w c
        shape = x.shape
        # collapse all spatial dimension
        x = x.reshape(shape[0], -1, shape[-1])
        mean = x.mean(dim=-2, keepdim=True)
        var = x.var(dim=-2, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            x = x * self.weight + self.bias
        # restore the spatial dimension
        x = x.reshape(shape)
        return x


def get_time_embedding(t, dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(t.shape) == 1

    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=t.device)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


# below code are taken from the amazing Hyena repo:
# https://github.com/HazyResearch/safari/blob/9ecfaf0e49630b5913fce19adec231b41c2e0e39/src/models/sequence/hyena.py#L64

class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 seq_len: int,
                 lr_pos_emb: float = 1e-5,
                 **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        assert emb_dim > 1
        bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]  # 1, 1, emb_dim
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register_parameter("z", nn.Parameter(z))
        optim = {"lr": lr_pos_emb}
        setattr(getattr(self, "z"), "_optim", optim)
        self.register_buffer("t", t)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


# reference convolution with residual connection
def fftconv_ref(u, k, D, dropout_mask, gelu=False, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)   # bias term
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, 'b H -> b H 1')).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


class ExponentialModulation(nn.Module):
    def __init__(
            self,
            d_model,
            fast_decay_pct=0.3,
            slow_decay_pct=1.5,
            target=1e-2,
            modulate: bool = True,
            shift: float = 0.0,
            **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register_buffer("deltas", deltas)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class GaussianFourierFeatureTransform(nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels,
                 mapping_size=256, scale=10, learnable=False,
                 num_heads=1):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size

        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size * num_heads)) * scale,
                               requires_grad=learnable)
        self.num_heads = num_heads

    def forward(self, x, unfold_head=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        if unfold_head:
            x = rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class LowRankKernel(nn.Module):
    # low rank kernel, ideally operates only on one dimension
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 positional_embedding='rotary',
                 pos_dim=1,
                 normalize=False,
                 softmax=False,
                 residual=True,
                 dropout=0,
                 scaling=1,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dim_head = dim_head
        self.heads = heads
        self.normalize = normalize
        self.residual = residual
        if dropout > 1e-6:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.to_q = nn.Linear(dim, dim_head*heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head*heads, bias=False)

        assert positional_embedding in ['rff', 'rotary', 'learnable', 'none']
        self.positional_embedding = positional_embedding
        self.pos_dim = pos_dim

        if positional_embedding == 'rff':
            self.pos_emb = GaussianFourierFeatureTransform(pos_dim, dim_head, scale=1,
                                                           learnable=False, num_heads=heads)
        elif positional_embedding == 'rotary':
            self.pos_emb = RotaryEmbedding(dim_head//self.pos_dim, min_freq=1/64)
        elif positional_embedding == 'learnable':
            self.pos_emb = nn.Sequential(
                GaussianFourierFeatureTransform(pos_dim, dim_head * heads // 2, scale=1,
                                                learnable=True),
                nn.Linear(dim_head * heads // 2, dim_head*heads, bias=False),
                nn.GELU(),
                nn.Linear(dim_head*heads, dim_head*heads, bias=False))
        else:
            pass
        self.init_gain = 0.02   # 1 / np.sqrt(dim_head)
        # self.diagonal_weight = nn.Parameter(1 / np.sqrt(dim_head) *
        #                                     torch.ones(heads, 1, 1), requires_grad=True)
        self.initialize_qk_weights()
        self.softmax = softmax

        self.residual = residual
        if self.residual:
            self.gamma = nn.Parameter(torch.tensor(1 / np.sqrt(dim_head)), requires_grad=True)
        else:
            self.gamma = 0
        self.scaling = scaling

    def initialize_qk_weights(self):
        xavier_uniform_(self.to_q.weight, gain=self.init_gain)
        xavier_uniform_(self.to_k.weight, gain=self.init_gain)
        # torch.nn.init.normal_(self.to_q.weight, std=self.init_gain)
        # torch.nn.init.normal_(self.to_k.weight, std=self.init_gain)

    def normalize_wrt_domain(self, x):
        x = (x - x.mean(dim=-2, keepdim=True)) / (x.std(dim=-2, keepdim=True) + 1e-5)
        return x

    def forward(self, u_x, u_y=None, pos_x=None, pos_y=None):
        # u_x, u_y: b n c
        # u_x is from the first source
        # u_y is from the second source
        # pos: b n d
        if u_y is None:
            u_y = u_x

        n = u_y.shape[1]

        q = self.to_q(u_x)
        k = self.to_k(u_y)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        if self.normalize:
            q = self.normalize_wrt_domain(q)
            k = self.normalize_wrt_domain(k)

        if self.positional_embedding != 'none' and pos_x is None:
            raise ValueError('positional embedding is not none but pos is None')

        if self.positional_embedding != 'rotary' and \
                self.positional_embedding != 'none' and \
                self.positional_embedding != 'rff':
            pos_x_emb = self.pos_emb(pos_x)
            if pos_y is None:
                pos_y_emb = pos_x_emb
            else:
                pos_y_emb = self.pos_emb(pos_y)
            q = q * pos_x_emb
            k = k * pos_y_emb
        elif self.positional_embedding == 'rff':

            pos_x_emb = self.pos_emb(pos_x, unfold_head=True)
            if pos_y is None:
                pos_y_emb = pos_x_emb
            else:
                pos_y_emb = self.pos_emb(pos_y, unfold_head=True)

            # duplicate q, k
            q_ = torch.cat((q, q), dim=-1)
            k_ = torch.cat((k, k), dim=-1)
            q = q_ * pos_x_emb
            k = k_ * pos_y_emb

        elif self.positional_embedding == 'rotary':
            if self.pos_dim == 2:
                assert pos_x.shape[-1] == 2
                q_freqs_x = self.pos_emb.forward(pos_x[..., 0], q.device)
                q_freqs_y = self.pos_emb.forward(pos_x[..., 1], q.device)
                q_freqs_x = repeat(q_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                q_freqs_y = repeat(q_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                if pos_y is None:
                    k_freqs_x = q_freqs_x
                    k_freqs_y = q_freqs_y
                else:
                    k_freqs_x = self.pos_emb.forward(pos_y[..., 0], k.device)
                    k_freqs_y = self.pos_emb.forward(pos_y[..., 1], k.device)
                    k_freqs_x = repeat(k_freqs_x, 'b n d -> b h n d', h=k.shape[1])
                    k_freqs_y = repeat(k_freqs_y, 'b n d -> b h n d', h=k.shape[1])

                q = apply_2d_rotary_pos_emb(q, q_freqs_x, q_freqs_y)
                k = apply_2d_rotary_pos_emb(k, k_freqs_x, k_freqs_y)
            elif self.pos_dim == 1:
                assert pos_x.shape[-1] == 1

                q_freqs = self.pos_emb.forward(pos_x[..., 0], q.device).unsqueeze(0)
                q_freqs = repeat(q_freqs, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                if pos_y is None:
                    k_freqs = q_freqs
                else:
                    k_freqs = self.pos_emb.forward(pos_y[..., 0], k.device).unsqueeze(0)
                    k_freqs = repeat(k_freqs, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                q = apply_rotary_pos_emb(q, q_freqs)
                k = apply_rotary_pos_emb(k, k_freqs)
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')
        else:  # do nothing
            pass

        K = torch.einsum('bhid,bhjd->bhij', q, k) * self.scaling  # if not on uniform grid, need to consider quadrature weights
        K = self.dropout(K)
        if self.softmax:
            K = F.softmax(K, dim=-1)
        if self.residual:
            K = K + self.gamma * torch.eye(n).to(q.device).view(1, 1, n, n) / n
        return K


class PoolingReducer(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super().__init__()
        self.to_in = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_ffn = PreNorm(in_dim, MLP([hidden_dim, hidden_dim, out_dim], GeAct(nn.GELU())))

    def forward(self, x):
        # note that the dimension to be pooled will be the last dimension
        # x: b nx ... c
        x = self.to_in(x)
        # pool all spatial dimension but the first one
        ndim = len(x.shape)
        x = x.mean(dim=tuple(range(2, ndim-1)))
        x = self.out_ffn(x)
        return x  # b nx c


class FABlock2D(nn.Module):
    # contains factorization and attention on each axis
    def __init__(self,
                 dim,
                 dim_head,
                 latent_dim,
                 heads,
                 dim_out,
                 use_rope=True,
                 kernel_multiplier=3,
                 scaling_factor=1.0):
        super().__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.dim_head = dim_head
        self.in_norm = nn.LayerNorm(dim)
        self.to_v = nn.Linear(self.dim, heads * dim_head, bias=False)
        self.to_in = nn.Linear(self.dim, self.dim, bias=False)

        self.to_x = nn.Sequential(
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )
        self.to_y = nn.Sequential(
            Rearrange('b nx ny c -> b ny nx c'),
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )

        positional_encoding = 'rotary' if use_rope else 'none'
        use_softmax = False
        self.low_rank_kernel_x = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,  # add a diagonal bias
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)
        self.low_rank_kernel_y = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)

        self.to_out = nn.Sequential(
            GroupNorm(heads, dim_head * heads, domain_wise=True, affine=False),
            nn.Linear(dim_head * heads, dim_out, bias=False),
            nn.GELU(),
            nn.Linear(dim_out, dim_out, bias=False))

    def forward(self, u, pos_lst):
        # x: b c h w
        u = self.in_norm(u)
        v = self.to_v(u)
        u = self.to_in(u)

        u_x = self.to_x(u)
        u_y = self.to_y(u)

        pos_x, pos_y = pos_lst
        k_x = self.low_rank_kernel_x(u_x, pos_x=pos_x)
        k_y = self.low_rank_kernel_y(u_y, pos_x=pos_y)

        u_phi = rearrange(v, 'b i l (h c) -> b h i l c', h=self.heads)
        u_phi = torch.einsum('bhij,bhjmc->bhimc', k_x, u_phi)
        u_phi = torch.einsum('bhlm,bhimc->bhilc', k_y, u_phi)
        u_phi = rearrange(u_phi, 'b h i l c -> b i l (h c)', h=self.heads)
        return self.to_out(u_phi)

