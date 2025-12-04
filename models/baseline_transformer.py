import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return x + self.positional_embedding(positions)

class TransformerBaseline(nn.Module):
    def __init__(
        self, 
        num_features, 
        num_targets, 
        d_model = 128, 
        n_heads = 4, 
        num_layers = 4, 
        dim_feedforward = 256, 
        dropout = 0.1,
        max_len = 256,
        pooling = "last"
    ):
        super().__init__()

        self.pooling = pooling
        #https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.input_proj = nn.Linear(num_features, d_model)

        self.positional_embedding = PositionalEmbedding(max_len = max_len, d_model = d_model)

        #https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = n_heads, 
            dim_feedforward = dim_feedforward, 
            dropout = dropout,
            batch_first = True,
            activation = "gelu")
        
        #https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        #https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.head = nn.Linear(d_model, num_targets)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.positional_embedding(x)
        x = self.encoder(x)
        if self.pooling == "last":
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)
        x = self.layer_norm(x)
        x = self.head(x)
        return x
        

        
        

        
        