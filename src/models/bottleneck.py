import torch
import torch.nn as nn

class LatentAligner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None):
        """
        An MLP to map the hidden states of Model A to Model B.
        """
        super().__init__()
        if hidden_dim is None:
            # Default to an intermediate size if not provided
            hidden_dim = (input_dim + output_dim) // 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        returns: [batch_size, seq_len, output_dim]
        """
        return self.mlp(x)
