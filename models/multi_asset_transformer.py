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

class MultiAssetTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        num_assets,
        d_model = 128,
        n_heads = 4,
        num_layers = 4,
        dim_feedforward = 256,
        dropout = 0.1,
        max_len = 256,
        pooling = "last",
        asset_embedding = True
    ):
        super().__init__()
        self.asset_embedding = asset_embedding
        if asset_embedding:
            #https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            self.asset_embed = nn.Embedding(num_assets, d_model)

        self.pooling = pooling
        self.num_assets = num_assets
        self.pos_embedding = PositionalEmbedding(max_len = max_len, d_model = d_model)
        self.input_proj = nn.Linear(num_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = n_heads, 
            dim_feedforward = dim_feedforward, 
            dropout = dropout,
            batch_first = True,
            activation = "gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B,A,W,F = x.shape
        x = x.view(B*A,W,F)
        x = self.input_proj(x)

        if self.asset_embedding:
            asset_ids = torch.arange(A, device=x.device).unsqueeze(0).expand(B, A).reshape(-1)
            asset_embed = self.asset_embed(asset_ids)
            x = x + asset_embed.unsqueeze(1)
        x = self.pos_embedding(x)
        x = self.encoder(x)
        if self.pooling == "last":
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)
        x = self.layer_norm(x)
        x = self.head(x).squeeze(-1)
        x = x.view(B,A)
        return x 
        

        

        
        
        