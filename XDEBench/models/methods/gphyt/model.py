"""
Reference:
    Towards a Physics Foundation Model. https://arxiv.org/abs/2509.13805
    Source: https://github.com/FloWsnr/General-Physics-Transformer/tree/main/gphyt/model
"""
from typing import Optional, Literal

import torch
import torch.nn as nn

from .attention import AttentionBlock
from .pos_encodings import (
    RotaryPositionalEmbedding,
    AbsPositionalEmbedding,
)
from models.base.tokenizer.tokenizer import Tokenizer, Detokenizer
from .derivatives import FiniteDifference
from .num_integration import Euler, RK4, Heun



class PhysicsTransformer(nn.Module):
    """
    Physics Transformer model.

    Parameters
    ----------
    ################################################################
    ########### Transformer parameters #############################
    ################################################################

    num_fields: int
        Number of input fields (physical fields).
    hidden_dim: int
        Hidden dimension inside the attention blocks.
        Should be divisible by 6 if Rope positional encoding is used.
    mlp_dim: int
        Hidden dimension inside the MLP.
    num_heads: int
        Number of attention heads.
    num_layers: int
        Number of attention blocks.
    pos_enc_mode: Literal["rope", "absolute"] = "rope"
        Position encoding mode. Can be "rope" or "absolute".
    patch_size: tuple[int, int, int]
        Patch size for spatial-temporal embeddings. (time, height, width)
    att_mode: Literal["full"] = "full"
        Attention mode. Can be "full".
    integrator: str
        Integrator to use
    img_size: tuple[int, int, int]
        Incoming image size (time, height, width)
    use_derivatives: bool, optional
        Whether to use derivatives in the model.

    ################################################################
    ########### Tokenizer parameters ###############################
    ################################################################

    tokenizer_mode: Literal["linear", "conv_net"] = "linear"
        Tokenizer mode. Can be "linear" or "conv_net".
    detokenizer_mode: Literal["linear", "conv_net"] = "linear"
        Detokenizer mode. Can be "linear" or "conv_net".
    tokenizer_net_channels: list[int] = None
        Number of channels in the tokenizer conv_net.
    detokenizer_net_channels: list[int] = None
        Number of channels in the detokenizer conv_net.
    tokenizer_overlap: int = 0
        Number of pixels to overlap between patches for the tokenizer.
    detokenizer_overlap: int = 0
        Number of pixels to overlap between patches for the detokenizer.

    ################################################################
    ########### Training parameters ################################
    ################################################################

    dropout: float = 0.0
        Dropout rate.
    stochastic_depth_rate: float = 0.0
        Stochastic depth rate.
    """

    def __init__(self, args):
        super().__init__()

        # differentiate between actual fields and input channels, which can be more due to derivatives
        n_dim = len(args.shape_list)
        num_input_channels = args.in_dim - n_dim 
        self.num_fields = args.in_dim - n_dim
        self.att_mode = "full"
        integrator = "Euler"
        
        img_size = args.shape_list
        patch_size = args.modes
        hidden_dim = args.width
        tokenizer_mode = "linear"
        detokenizer_mode = "linear"
        tokenizer_net_channels = None
        detokenizer_net_channels = None
        tokenizer_overlap = 0
        detokenizer_overlap = 0
        pos_enc_mode = "absolute"
        mlp_dim = args.width * 4
        num_heads = args.n_heads
        dropout = 0.0
        stochastic_depth_rate = 0.0
        num_layers = args.n_layers

        n_patch_t = img_size[0] // patch_size[0]
        n_patch_h = img_size[1] // patch_size[1]
        n_patch_w = img_size[2] // patch_size[2]

        # Initialize derivatives module
        self.use_derivatives = False
        if self.use_derivatives:
            self.derivatives = FiniteDifference(
                num_channels=self.num_fields, filter_1d="2nd"
            )
            # if derivatives are used, the input channels are multiplied by 4 (original, dt, dh, dw)
            # however, the output channels of the tokenizer are still the original input channels
            num_input_channels *= 4

        if integrator == "Euler":
            self.integrator = Euler()
        elif integrator == "RK4":
            self.integrator = RK4()
        elif integrator == "Heun":
            self.integrator = Heun()
        else:
            self.integrator = None

        self.tokenizer = Tokenizer(
            patch_size=patch_size,
            in_channels=num_input_channels,
            dim_embed=hidden_dim,
            mode=tokenizer_mode,
            conv_net_channels=tokenizer_net_channels,
            overlap=tokenizer_overlap,
        )

        # Initialize positional encodings
        if pos_enc_mode == "rope":
            att_pos_encodings = RotaryPositionalEmbedding(dim=hidden_dim, base=10000)
            self.init_pos_encodings = None
        elif pos_enc_mode == "absolute":
            self.init_pos_encodings = AbsPositionalEmbedding(
                num_channels=hidden_dim,
                time=n_patch_t,
                height=n_patch_h,
                width=n_patch_w,
            )
            att_pos_encodings = None
        else:
            raise ValueError(f"Invalid positional encoding mode: {pos_enc_mode}")

        # Initialize attention blocks
        self.attention_blocks = nn.Sequential(
            *[
                AttentionBlock(
                    att_type=self.att_mode,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    time=n_patch_t,
                    height=n_patch_h,
                    width=n_patch_w,
                    dropout=dropout,
                    stochastic_depth_rate=stochastic_depth_rate,
                    pe=att_pos_encodings,
                )
                for _ in range(num_layers)
            ]
        )

        # Initialize tokenizer and detokenizer

        self.detokenizer = Detokenizer(
            patch_size=patch_size,
            dim_embed=hidden_dim,
            out_channels=self.num_fields,  # important to set to num_fields
            mode=detokenizer_mode,
            conv_net_channels=detokenizer_net_channels,
            overlap=detokenizer_overlap,
            img_size=img_size,
        )

    def differentiate(self, x: torch.Tensor) -> torch.Tensor:
        assert not torch.isnan(x).any(), "Input contains NaNs"

        if self.use_derivatives:
            dt, dh, dw = self.derivatives(x)
            x = torch.cat([x, dt, dh, dw], dim=-1)

        # Split into patches
        x = self.tokenizer(x)
        if self.init_pos_encodings is not None:
            x = self.init_pos_encodings(x)

        # Apply N attention blocks (norm, att, norm, mlp)
        x = self.attention_blocks(x)

        # # Apply de-patching
        x = self.detokenizer(x)
        return x

    def forward(self, x, coords):
        x = x.permute(0, 2, 3, 4, 1) # [batchSize, nx, ny, nz, in_channels]
        if self.integrator is None:
            out = self.differentiate(x)
        else:
            out = self.integrator(self.differentiate, x)
        out = out.permute(0, 4, 1, 2, 3)
        return out
