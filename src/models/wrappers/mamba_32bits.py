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

    def __init__( self,
        
        d_model: int = 64,
        num_layer_encoder = 1, 
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
        self.residual_scale = residual_scale
        

        self.discrete_embedding = nn.Embedding(2, self.d_model) # for frozen 
        self.linear_embedding1 = nn.Linear(in_features=1, out_features=d_model )
     #   self.linear_embedding2 = nn.Linear(in_features=1, out_features=d_model)

        self.linear_input_layer = nn.Linear(2*self.d_model, d_model)

        self.alpha = nn.Parameter(torch.tensor(1.0))   # for channel
        self.beta = nn.Parameter(torch.tensor(1.0))    # for SNR
        self.gamma = nn.Parameter(torch.tensor(1.0))   # for frozen

        self.encoder_layers = nn.ModuleList([
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

       
      #  nn.init.xavier_uniform_(self.linear_embedding2.weight)
     #   if self.linear_embedding2.bias is not None:
     #       nn.init.zeros_(self.linear_embedding2.bias)

       
        nn.init.normal_(self.discrete_embedding.weight, mean=0.0, std=1e-2)

     
        nn.init.xavier_uniform_(self.final_proj_layer.weight)
        if self.final_proj_layer.bias is not None:
            nn.init.zeros_(self.final_proj_layer.bias)

       
        if hasattr(self.layer_norm, 'weight'):
            nn.init.ones_(self.layer_norm.weight)
        if hasattr(self.layer_norm, 'bias'):
            nn.init.zeros_(self.layer_norm.bias)

       
        #with torch.no_grad():
        #    self.alpha.fill_(1.0)     # observation — usually strong
        #    self.beta.fill_(0.05)     # SNR — starting small so it doesn't dominate
        #    self.gamma.fill_(0.5)     # prior — small but present
        

    
    def forward(self, channel_ob_vector, frozen_prior, SNR_db):

        if channel_ob_vector.dim()!=2 or frozen_prior.dim()!=2:
            raise ValueError("Channel observation vector and frozen prior vector must be (Batch,Sequence length)")
        
        ch_emb = self.linear_embedding1(channel_ob_vector.unsqueeze(-1))
      #  snr_emb = self.linear_embedding2(SNR_db.unsqueeze(-1).float())
        froz_emb = self.discrete_embedding(frozen_prior)

        

      #  snr_emb = snr_emb.unsqueeze(1)
      #  snr_emb = snr_emb.expand(-1, 32, -1) # to make sure for each bit's d_model embedding, there is a single d_model embedding value of snr

      #  print(f"channel vector emb shape: {ch_emb.shape}\n")
       # print(f"snr emb shape: {snr_emb.shape}\n")
     #   print(f"frozen shape: {froz_emb.shape}")
      

        #encoder_input = self.alpha*ch_emb+self.beta*snr_emb+self.gamma*froz_emb #ramro result ayena vane try concatenation without parameters multiply

      #  print("check 1")
        encoder_input = torch.cat([ch_emb, froz_emb], dim=-1)
        encoder_input = self.linear_input_layer(encoder_input)

        
     #   residuals = []
        x = encoder_input
        for idx, layer in enumerate(self.encoder_layers):
            x_new = layer(x)
        #    residuals.append(x_new)
            x = x_new*self.residual_scale + x
            x = self.post_norms[idx](x)
        
        #if len(residuals) > 1:
      #   x = sum(residuals) / len(residuals)
       
        return self.final_proj_layer(x).squeeze(-1)
    


        
        
    


# testing for N=3, batch=1

'''
model = MambaPolarDecoder(d_model=4, seq_len=3).to('cuda')

channel_ob_vector = torch.tensor([ [1.1, 0, 2], ]).float().to('cuda')
frozen_prior_vector = torch.tensor([ [0, 0, 1], ]).int().to('cuda')
snr = torch.tensor([[6]]).float().to('cuda')

print(channel_ob_vector.shape)

output = model(channel_ob_vector, frozen_prior_vector, snr)
print(output)

'''     

        

        











