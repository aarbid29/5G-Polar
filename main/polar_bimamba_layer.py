import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import math
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

# from main.helpers.config import Code
from dataclasses import dataclass
from typing import Any

@dataclass
class Code:
    n: int
    k: int
    code_type: str
    pc_matrix: Any = None
    generator_matrix: Any = None


def mask_larger_matrix(M, mask):
    """Apply mask to selective scan matrices"""
    M[:, :mask.size(0), :mask.size(1)][mask.expand(M.size(0), mask.size(0), mask.size(1))] = 0.0
    M[:, mask.size(0):, mask.size(1):] = 0
    return M


def apply_mask_to_ssm(A, B, C, D, mask):
    """Apply parity check mask to SSM parameters"""
    B = mask_larger_matrix(B, mask)
    C = mask_larger_matrix(C, mask)
    return A, B, C, D


class MaskedMambaLayer(nn.Module):
    """
    Mamba layer with parity check masking for polar codes.

    Takes noisy embeddings
    Applies Mamba SSM
    Uses the parity-check mask
    Produces updated beliefs
    
    """
    def __init__(
        self,
        code: Code,
        d_model: int = 128,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: int = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.code_length = code.n
        self.syndrome_length = code.pc_matrix.size(0) if code.pc_matrix is not None else 0
        self.output_length = self.code_length + self.syndrome_length
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # SSM projections
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, pc_mask):
        """
        hidden_states: (B, L, D)
        pc_mask: (L, syndrome_length) - parity check mask
        Returns: (B, L, D)
        """
        batch, seqlen, dim = hidden_states.shape

        # Input projection and rearrange
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        x, z = xz.chunk(2, dim=1)
        
        # Causal convolution
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # SSM parameters
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        # Apply mask to SSM parameters
        D = self.D.float()
        A, B, C, D = apply_mask_to_ssm(A, B, C, D, pc_mask)
        
        # Selective scan
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            D,
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        
        y = rearrange(y, "b d l -> b l d")
        
        # Apply output mask
        mask_larger_matrix(y, pc_mask)
        
        out = self.out_proj(y)
        return out


class BiMambaLayer(nn.Module):
    """
    Bidirectional Mamba layer with forward and reverse processing.
    """
    def __init__(self, code: Code, d_model: int = 128, d_state: int = 128, 
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        
        self.syndrome_length = code.pc_matrix.size(0) if code.pc_matrix is not None else 0
        self.n = code.n
        
        # Forward and reverse Mamba layers
        self.mamba_forward = MaskedMambaLayer(
            code=code,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        self.mamba_reverse = MaskedMambaLayer(
            code=code,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, emb, pc_mask):
        """
        emb: (B, L, D)
        pc_mask: (L, syndrome_length)
        """
        # Forward pass
        mamba_forward_out = self.mamba_forward(emb, pc_mask)
        
        # Reverse pass
        emb_reversed = torch.flip(emb, [1])
        pc_mask_reversed = torch.flip(pc_mask, [1])
        mamba_reverse_out = self.mamba_reverse(emb_reversed, pc_mask_reversed)
        mamba_reverse_out = torch.flip(mamba_reverse_out, [1])
        
        # Combine forward and reverse
        mamba_out = mamba_forward_out + mamba_reverse_out
        
        return self.norm(mamba_out)
