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

class CrossAssetTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        num_assets,
        d_model = 128,
        n_heads = 4,
        num_temporal_layers = 4,
        num_cross_asset_layers = 4,
        dim_feedforward = 256,
        dropout = 0.1,
        max_len = 256,
        pooling = "last",
    ):
        super().__init__()
        self.num_assets = num_assets
        self.d_model = d_model
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_embedding = PositionalEmbedding(max_len = max_len, d_model = d_model)
        
        temporal_layer = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = n_heads, 
            dim_feedforward = dim_feedforward, 
            dropout = dropout,
            batch_first = True,
            activation = "gelu"
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers = num_temporal_layers)
        
        self.pooling = pooling  
        self.asset_embedding = nn.Embedding(num_assets, d_model)

        cross_asset_layer = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = n_heads, 
            dim_feedforward = dim_feedforward, 
            dropout = dropout,
            batch_first = True,
            activation = "gelu"
        )
        self.cross_asset_encoder = nn.TransformerEncoder(cross_asset_layer, num_layers = num_cross_asset_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        B,A,W,F = x.shape
        x = x.view(B*A,W,F)
        x = self.input_proj(x)
        x = self.pos_embedding(x)
        x= self.temporal_encoder(x)
        if self.pooling == "last":
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)
        x = x.view(B,A,self.d_model)

        asset_indices= torch.arange(self.num_assets, device=x.device).unsqueeze(0)
        asset_embed = self.asset_embedding(asset_indices)
        x = x + asset_embed
        x = self.cross_asset_encoder(x)
        x = self.layer_norm(x)
        x = self.head(x).squeeze(-1)
        return x    
        