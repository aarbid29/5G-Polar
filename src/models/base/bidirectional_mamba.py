import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BiMambaBlock(nn.Module):
    # BiMambaBlock:
    # Hastwo branches: forward and reverse.
    #Each branch uses pre-LN, Mamba, dropout and residual.
    # Applies Mamba operations, followed by Feed-Forward Networks, with residual connections and Layer Normalization.
    # The final output is the average of the forward and reverse branches.
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        share_norm: bool = False,
        share_ffn: bool = False,
    ):
        super().__init__()
        # forward branch
        self.pre_ln_f = nn.LayerNorm(d_model)
        self.mamba_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.post_ln_f = nn.LayerNorm(d_model)
        self.ffn_f = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # reverse branch
        if share_norm:
            self.pre_ln_r = self.pre_ln_f
            self.post_ln_r = self.post_ln_f
        else:
            self.pre_ln_r = nn.LayerNorm(d_model)
            self.post_ln_r = nn.LayerNorm(d_model)

        self.mamba_r = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        if share_ffn:
            self.ffn_r = self.ffn_f
        else:
            self.ffn_r = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )

        self.dropout = nn.Dropout(dropout)
        self.residual_scale = residual_scale

    def forward_branch(self, x, pre_ln, mamba, post_ln, ffn, flip_time=False):
       # x: (B, S, D)
     #flip_time: if True, flip along time dim before applying mamba and unflip after
        x_in = x
        if flip_time:
            x_proc = torch.flip(x, dims=[1])
        else:
            x_proc = x

        h = pre_ln(x_proc)
        h = mamba(h)
        h = self.dropout(h)
        if flip_time:
            h = torch.flip(h, dims=[1])

        # residual + post norm
        h = x_in + self.residual_scale * h
        h = post_ln(h)

        # feedforward with residual
        y = ffn(h)
        y = self.dropout(y)
        y = h + self.residual_scale * y
        y = post_ln(y)  # apply post norm again 
        return y

    def forward(self, x):
        out_f = self.forward_branch(x, self.pre_ln_f, self.mamba_f, self.post_ln_f, self.ffn_f, flip_time=False)
        out_r = self.forward_branch(x, self.pre_ln_r, self.mamba_r, self.post_ln_r, self.ffn_r, flip_time=True)
        return 0.5 * (out_f + out_r)


class BiMambaEncoder(nn.Module):
    # BiMambaEncoder:
    # Takes an input sequence, adds positional embeddings, and processes the sequence through multiple BiMambaBlocks.
    #Stacks num_layers BiMambaBlock.
    # Outputs the transformed sequence with shape (B, S, d_model).
  
    def __init__(
        self,
        d_model: int = 64,
        num_layers: int = 4,
        seq_len: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        share_norm: bool = False,
        share_ffn: bool = False,
        use_embedding_for_bits: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.use_embedding_for_bits = use_embedding_for_bits

    
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))

        self.layers = nn.ModuleList([
            BiMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                residual_scale=residual_scale,
                share_norm=share_norm,
                share_ffn=share_ffn,
            ) for _ in range(num_layers)
        ])

        
        self.norm = nn.LayerNorm(d_model)
      
        self._init_weights()

    def _init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
       #x is (B,S,D) here
                        
        if x.dim() == 3:
            if x.size(2) != self.d_model:
                raise ValueError(f"Input last dim {x.size(2)} != d_model {self.d_model}")
            h = x
        else:
            raise ValueError("Input must be (B,S,  D) i.e already embedded") 

        # add positional embeddings 
        L = h.size(1)
        if L > self.seq_len:
            raise ValueError(f"Sequence length {L} > seq_len {self.seq_len}")
        h = h + self.pos_emb[:, :L, :]

        # pass through stacked BiMambaBlocks
        for layer in self.layers:
            h = layer(h)

        return self.norm(h)
