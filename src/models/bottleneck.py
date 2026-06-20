import torch
import torch.nn as nn

class LatentAligner(nn.Module):

    def __init__(self, dim_a=896, dim_b=2048):
        super().__init__()
        self.proj = nn.Linear(dim_a, dim_b)
        
        # Deep Residual MLP for token-by-token mapping
        self.ffn = nn.Sequential(
            nn.Linear(dim_b, dim_b * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_b * 4, dim_b)
        )
        self.norm1 = nn.LayerNorm(dim_b)
        self.norm2 = nn.LayerNorm(dim_b)

    def forward(self, x):
        # x is [batch_size, seq_len, dim_a]
        x = self.proj(x)
        x = self.norm1(x)
        
        # Residual connection
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        
        return out # [batch_size, seq_len, dim_b]
