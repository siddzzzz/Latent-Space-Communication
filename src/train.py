import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import HiddenStateDataset, get_training_data
from models.bottleneck import LatentAligner
from aligner import AlignerLoss

# Hyperparameters
MODEL_A = "Qwen/Qwen2.5-0.5B"
MODEL_B = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_SAMPLES = 500 # Small number for quick local testing

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training using device: {device}")

    # 1. Initialize Pipeline (Loads Models)
    pipeline = HiddenStateDataset(model_a_name=MODEL_A, model_b_name=MODEL_B)

    # 2. Get Data
    texts = get_training_data(num_samples=NUM_SAMPLES)
    
    # Create a simple dataloader for batching text
    dataloader = DataLoader(texts, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Aligner
    dim_a = pipeline.model_a.config.hidden_size
    dim_b = pipeline.model_b.config.hidden_size
    
    print(f"Model A hidden dim: {dim_a}")
    print(f"Model B hidden dim: {dim_b}")

    aligner = LatentAligner(input_dim=dim_a, output_dim=dim_b).to(device)
    loss_fn = AlignerLoss(alpha=0.5).to(device)
    optimizer = torch.optim.AdamW(aligner.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    aligner.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_abs = 0
        total_rel = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_texts in progress_bar:
            # Get ground truth hidden states (mean-pooled to [batch_size, hidden_dim])
            # We don't need gradients for this part
            with torch.no_grad():
                h_a, h_b = pipeline.get_hidden_states(batch_texts)

            # Map Model A's latent space to Model B's
            mapped_a = aligner(h_a)

            # Calculate Loss
            optimizer.zero_grad()
            loss, abs_loss, rel_loss = loss_fn(mapped_a, h_b)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_abs += abs_loss.item()
            total_rel += rel_loss.item()
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "abs": f"{abs_loss.item():.4f}",
                "rel": f"{rel_loss.item():.4f}"
            })

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f} (Abs: {total_abs/len(dataloader):.4f}, Rel: {total_rel/len(dataloader):.4f})")

    # 5. Save the trained aligner
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(aligner.state_dict(), "./checkpoints/aligner.pth")
    print("Training complete! Aligner saved to ./checkpoints/aligner.pth")

if __name__ == "__main__":
    train()
