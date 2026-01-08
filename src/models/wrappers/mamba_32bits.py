import torch, torch.nn as nn
from models.base.bidirectional_mamba import BiMambaBlock, BiMambaEncoder

class MambaPolarDecoder(nn.Module):

    """
    A decoder for Polar Codes of block length N based on Bidirectional Mamba.

    Takes: Channel Observation Vector(N), Frozen Bit Prior Vector(N), SNR(single value, in db)
    Input shape: (batch_size, block_length, 3) -> includes channel_output_value, frozen_prior_value, snr

    Predicts: Channel Input Vector(N)
    Output shape: (batch_size, blocklength) -> raw logits representing predicted bits
    """

    def __init__(self,
                 d_model: int = 64,
                 num_layer_encoder=1,
                 num_layers_bimamba_block: int = 4,
                 seq_len: int = 512,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dropout: float = 0.1,
                 residual_scale: float = 1.0,
                 share_norm: bool = False,
                 share_ffn: bool = False):
        super().__init__()

        self.d_model = d_model
        self.num_layer_encoder = num_layer_encoder
        self.num_layers_bimamba_block = num_layers_bimamba_block
        self.seq_len = seq_len
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout
        self.residual_scale = residual_scale

        self.discrete_embedding = nn.Embedding(2, self.d_model)  # for frozen
        self.linear_embedding1 = nn.Linear(in_features=1, out_features=d_model)

        self.linear_input_layer = nn.Linear(2 * self.d_model, d_model)

        self.alpha = nn.Parameter(torch.tensor(1.0))  # for channel
        self.beta = nn.Parameter(torch.tensor(1.0))  # for SNR
        self.gamma = nn.Parameter(torch.tensor(1.0))  # for frozen

        self.encoder_layers = nn.ModuleList([
            #bimamba encode model
            BiMambaEncoder(
                d_model=self.d_model,
                num_layers=self.num_layers_bimamba_block,
                seq_len=self.seq_len,
                d_state=d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout,
            ) for _ in range(self.num_layer_encoder)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        self.post_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.num_layer_encoder)
        ])
        self.final_proj_layer = nn.Linear(d_model, 1)

        self.init_weights_new()

    def init_weights_new(self):
        nn.init.xavier_uniform_(self.linear_embedding1.weight)
        if self.linear_embedding1.bias is not None:
            nn.init.zeros_(self.linear_embedding1.bias)

        nn.init.normal_(self.discrete_embedding.weight, mean=0.0, std=1e-2)

        nn.init.xavier_uniform_(self.final_proj_layer.weight)
        if self.final_proj_layer.bias is not None:
            nn.init.zeros_(self.final_proj_layer.bias)

        if hasattr(self.layer_norm, 'weight'):
            nn.init.ones_(self.layer_norm.weight)
        if hasattr(self.layer_norm, 'bias'):
            nn.init.zeros_(self.layer_norm.bias)

    def forward(self, channel_ob_vector, frozen_prior, SNR_db):
        if channel_ob_vector.dim() != 2 or frozen_prior.dim() != 2:
            raise ValueError("Channel observation vector and frozen prior vector must be (Batch, Sequence length)")

        ch_emb = self.linear_embedding1(channel_ob_vector.unsqueeze(-1))  # channel embedding
        froz_emb = self.discrete_embedding(frozen_prior)  # frozen embedding

        # concatenate and passed to linear layer
        encoder_input = torch.cat([ch_emb, froz_emb], dim=-1)
        encoder_input = self.linear_input_layer(encoder_input)

        x = encoder_input
        for idx, layer in enumerate(self.encoder_layers):
            x_new = layer(x)
            x = x_new * self.residual_scale + x
            x = self.post_norms[idx](x)

        return self.final_proj_layer(x).squeeze(-1)


