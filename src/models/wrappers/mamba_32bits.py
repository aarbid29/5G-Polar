import torch
import torch.nn as nn
from models.base.bidirectional_mamba import BiMambaBlock, BiMambaEncoder


class MambaPolarDecoder(nn.Module):
    """
    A decoder for Polar Codes of block length N based on Bidirectional Mamba.

    Takes:
      - Channel Observation Vector (N)
      - Frozen Bit Prior Vector (N)
      - SNR (single value, in dB)

    Input shape:
      channel_ob_vector : (batch_size, block_length)
      frozen_prior      : (batch_size, block_length)

    Output:
      (batch_size, block_length) -> raw logits
    """

    def __init__(
        self,
        d_model: int = 64,
        num_layer_encoder: int = 1,
        num_layers_bimamba_block: int = 4,
        seq_len: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        share_norm: bool = False,
        share_ffn: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layer_encoder = num_layer_encoder
        self.num_layers_bimamba_block = num_layers_bimamba_block
        self.seq_len = seq_len
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout

        # learnable residual scale
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale))

        # embeddings
        self.discrete_embedding = nn.Embedding(2, self.d_model)   # frozen bits
        self.linear_embedding1 = nn.Linear(1, self.d_model)      # channel values

        # input fusion (NONLINEAR – important)
        self.input_layer = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # BiMamba encoders
        self.encoder_layers = nn.ModuleList([
            BiMambaEncoder(
                d_model=self.d_model,
                num_layers=self.num_layers_bimamba_block,
                seq_len=self.seq_len,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout,
            )
            for _ in range(self.num_layer_encoder)
        ])

        # post-residual norms (ENABLED)
        self.post_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.num_layer_encoder)
        ])

        self.final_proj_layer = nn.Linear(self.d_model, 1)

        self.init_weights_new()

    def init_weights_new(self):
        nn.init.xavier_uniform_(self.linear_embedding1.weight)
        nn.init.zeros_(self.linear_embedding1.bias)

        nn.init.normal_(self.discrete_embedding.weight, mean=0.0, std=1e-2)

        nn.init.xavier_uniform_(self.final_proj_layer.weight)
        nn.init.zeros_(self.final_proj_layer.bias)

        for ln in self.post_norms:
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)

    def forward(self, channel_ob_vector, frozen_prior, SNR_db=None):
        if channel_ob_vector.dim() != 2 or frozen_prior.dim() != 2:
            raise ValueError(
                "channel_ob_vector and frozen_prior must be (batch, seq_len)"
            )

        # embeddings
        ch_emb = self.linear_embedding1(channel_ob_vector.unsqueeze(-1))
        froz_emb = self.discrete_embedding(frozen_prior)

        # fuse inputs
        x = torch.cat([ch_emb, froz_emb], dim=-1)
        x = self.input_layer(x)

        # BiMamba encoder stack
        for idx, layer in enumerate(self.encoder_layers):
            x_new = layer(x)
            x = x_new * self.residual_scale + x
            x = self.post_norms[idx](x)

        # bit logits
        return self.final_proj_layer(x).squeeze(-1)
