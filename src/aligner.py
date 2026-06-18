import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignerLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha: Weight for the relative representation loss.
               Loss = MSE + alpha * Relative_Loss
        """
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, mapped_a, target_b):
        """
        mapped_a: The output from the LatentAligner (mapped hidden state of Model A)
                  Shape: [batch_size, seq_len, dim_B]
        target_b: The true hidden state of Model B
                  Shape: [batch_size, seq_len, dim_B]
        """
        # Flatten batch and seq_len for distance computation
        # [N, dim_B] where N = batch_size * seq_len
        flat_a = mapped_a.view(-1, mapped_a.size(-1))
        flat_b = target_b.view(-1, target_b.size(-1))
        
        # 1. Absolute Loss (MSE)
        abs_loss = self.mse_loss(flat_a, flat_b)

        # 2. Relative Loss (Cosine Similarity of pairwise distances)
        # To avoid massive memory spikes, we compute cosine sim of a small random sample if N is large
        sample_size = min(flat_a.size(0), 1024)
        indices = torch.randperm(flat_a.size(0))[:sample_size]
        
        sampled_a = flat_a[indices]
        sampled_b = flat_b[indices]

        # Compute pairwise cosine similarity matrix for A and B
        # Normalize vectors to unit length
        norm_a = F.normalize(sampled_a, p=2, dim=1)
        norm_b = F.normalize(sampled_b, p=2, dim=1)

        # Cosine similarity matrices: [sample_size, sample_size]
        sim_a = torch.matmul(norm_a, norm_a.t())
        sim_b = torch.matmul(norm_b, norm_b.t())

        # The relative loss is the difference in geometric relationships
        rel_loss = self.mse_loss(sim_a, sim_b)

        total_loss = abs_loss + self.alpha * rel_loss

        return total_loss, abs_loss, rel_loss
