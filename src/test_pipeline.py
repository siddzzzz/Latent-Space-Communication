import torch
import torch.nn.functional as F
from dataset import HiddenStateDataset
from models.bottleneck import LatentAligner
from train import MODEL_A, MODEL_B

def test_communication(test_sentence: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}")

    # 1. Load the frozen LLMs
    pipeline = HiddenStateDataset(model_a_name=MODEL_A, model_b_name=MODEL_B)

    # 2. Load the trained Aligner
    dim_a = pipeline.model_a.config.hidden_size
    dim_b = pipeline.model_b.config.hidden_size
    
    aligner = LatentAligner(input_dim=dim_a, output_dim=dim_b).to(device)
    
    try:
        aligner.load_state_dict(torch.load("./checkpoints/aligner.pth", map_location=device))
        aligner.eval()
        print("Successfully loaded trained aligner!")
    except FileNotFoundError:
        print("WARNING: Could not find trained aligner checkpoint. Using untrained initialized weights for demonstration.")

    # 3. Get Hidden States
    print(f"\nProcessing sentence: '{test_sentence}'")
    with torch.no_grad():
        h_a, h_b = pipeline.get_hidden_states([test_sentence])
        
        # 4. Map Model A's thoughts to Model B's space
        mapped_a = aligner(h_a)

        # 5. Measure similarity
        # Cosine similarity between what Model B *actually* thinks, 
        # and what Model A *tried to tell* Model B.
        cos_sim = F.cosine_similarity(mapped_a, h_b)
        mse_dist = F.mse_loss(mapped_a, h_b)

        print("-" * 30)
        print("Results:")
        print(f"Cosine Similarity: {cos_sim.item():.4f} (1.0 is perfect alignment)")
        print(f"MSE Distance:      {mse_dist.item():.4f} (0.0 is perfect alignment)")
        print("-" * 30)

        if cos_sim.item() > 0.8:
            print("Excellent alignment! The models are communicating well.")
        elif cos_sim.item() > 0.5:
            print("Moderate alignment. The mapper is learning the concepts.")
        else:
            print("Poor alignment. The mapping requires more training or a better architecture.")

if __name__ == "__main__":
    test_sentence = "The integration of multi-agent architectures using continuous latent spaces represents a significant leap in AI efficiency."
    test_communication(test_sentence)
