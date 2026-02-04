import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from dataclasses import dataclass
from typing import Any
import copy
from polar_bimamba_layer import BiMambaLayer   # still external

@dataclass
class Code:
    """Code configuration for polar codes"""
    n: int
    k: int
    code_type: str
    pc_matrix: Any = None
    generator_matrix: Any = None


@dataclass
class BiMambaConfig:
    """Default Config for BiMamba Polar Decoder"""
    epochs: int = 20000
    workers: int = 4
    warmup_lr: float = 1e-3
    warmup_length: int = 10
    lr: float = 2.5e-4
    batch_size: int = 128
    test_batch_size: int = 512
    train_batch_count: int = 1000
    test_batch_count: int = 1000
    seed: int = 42
    eta_min: float = 1e-6
    gradient_clipping: float = 1.0
    zero_cw: bool = True
    T_max: int = 20000

    # Model architecture
    N_dec: int = 8
    d_model: int = 128
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    code: Code = None
    path: str = None
    experiment_type: str = None

    seq_len: int = 32

    enable_multi_loss: bool = True
    enable_early_stopping: bool = True
    early_stopping_patience: int = 3


def sign_to_bin(x):

    return 0.5 * (1 - x)


def bin_to_sign(x):
    return 1 - 2 * x


def build_pc_mask(code: Code):
    if code.pc_matrix is None:
        return torch.eye(code.n, code.n).bool()

    pc_mask = (code.pc_matrix > 0).T
    return pc_mask



class PolarBiMambaDecoder(nn.Module):
    """
    BiMamba decoder for polar codes (N=32).
        (noisy magnitudes + syndrome)
                
        learnable embedding
                
        BiMambaLayer x nDEc)
                
        linear heads
                
        soft bits (probabilities)
                
        hard decision + parity check
    """

    def __init__(self, config: BiMambaConfig):
        super().__init__()

        self.n = config.code.n
        self.syndrome_length = (
            config.code.pc_matrix.size(0) if config.code.pc_matrix is not None else 0
        )

        # Register parity check matrix
        if config.code.pc_matrix is not None:
            self.register_buffer("pc_matrix", config.code.pc_matrix.float())
        else:
            self.register_buffer("pc_matrix", torch.eye(self.n))

        # Mask
        pc_mask = build_pc_mask(config.code)
        self.register_buffer("pc_mask", pc_mask)

        # Learnable embedding
        self.src_embed = nn.Parameter(
            torch.ones((self.n + self.syndrome_length, config.d_model))
        )

        # BiMamba layers
        self.bimamba_layers = ModuleList(
            [
                BiMambaLayer(
                    code=config.code,
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                )
                for _ in range(config.N_dec)
            ]
        )

        # Output heads
        self.resize_output_dim = nn.Linear(config.d_model, 1)
        self.resize_output_length = nn.Linear(self.n + self.syndrome_length, self.n)

        self.enable_early_stopping = config.enable_early_stopping
        self.enable_multi_loss = config.enable_multi_loss

        # Init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _hidden_to_output(self, hidden):
        hidden = self.resize_output_dim(hidden)
        out = self.resize_output_length(hidden.squeeze(-1))
        return (1 - F.tanh(out)) / 2

    def forward(self, magnitude, syndrome):

        inp = torch.cat([magnitude, syndrome], -1) # Concatenate noisy magnitude + syndrome
        emb = self.src_embed.unsqueeze(0) * inp.unsqueeze(-1) # Multiply by learnable embedding

        hidden = emb  # goes into BiMambaLayer
        outputs = []

        for layer in self.bimamba_layers:
            hidden = layer(hidden, self.pc_mask)
            layer_out = self._hidden_to_output(hidden)

            if self.training or not self.enable_early_stopping:
                outputs.append(layer_out)
            else:
                outputs = [layer_out]

            if self.enable_early_stopping and self.is_corrected(layer_out, syndrome):
                break

        return outputs

    def is_corrected(self, z, syndrome):
        x = sign_to_bin(torch.sign(bin_to_sign(F.tanh(z))))
        mat = self.pc_matrix[None, :, :]
        mult = (mat @ x.unsqueeze(-1)).squeeze(-1)
        return torch.all(mult % 2 == sign_to_bin(syndrome))

    def loss(self, z_pred, z2, y):

        losses = []
        z2 = sign_to_bin(torch.sign(z2))

        for z in z_pred:
            if z.grad_fn is not None:
                losses.append(F.binary_cross_entropy(z, z2))

            x_pred = sign_to_bin(torch.sign(bin_to_sign(F.tanh(z)) * torch.sign(y)))

            mat = self.pc_matrix[None, :, :]
            mult = mat @ x_pred.unsqueeze(-1)

            if self.enable_early_stopping and torch.all(mult % 2 == 0):
                break

        total_loss = sum(losses) if losses else torch.tensor(0.0, device=z_pred[0].device)
        return total_loss, x_pred

    def get_codeword(self, z_pred, y):

        for z in z_pred:
            x_pred = sign_to_bin(torch.sign(bin_to_sign(z) * torch.sign(y)))
            mat = self.pc_matrix[None, :, :]
            mult = mat @ x_pred.unsqueeze(-1)
            if torch.all(mult % 2 == 0):
                break

        return x_pred
