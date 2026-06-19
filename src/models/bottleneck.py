import torch
import torch.nn as nn

class LatentAligner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None):
        """
        Maps a vector from Model A's latent space to Model B's embedding space.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: [batch_size, input_dim]
        returns: [batch_size, 1, output_dim] (Adds a sequence dimension for soft-prompting)
        """
        # Ensure input is 2D
        if x.dim() > 2:
            x = x.squeeze()
            if x.dim() == 1:
                x = x.unsqueeze(0)
                
        out = self.mlp(x)
        # Reshape to [batch_size, seq_len=1, output_dim]
        return out.unsqueeze(1)
