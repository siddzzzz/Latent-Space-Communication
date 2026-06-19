import torch
from dataset import LatentCommunicationPipeline
from models.bottleneck import LatentAligner
from train import MODEL_A, MODEL_B

def test_latent_prompting(test_sentence: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}")

    pipeline = LatentCommunicationPipeline(model_a_name=MODEL_A, model_b_name=MODEL_B)

    dim_a = pipeline.model_a.config.hidden_size
    dim_b = pipeline.model_b.config.hidden_size
    
    aligner = LatentAligner(input_dim=dim_a, output_dim=dim_b).to(device)
    
    try:
        aligner.load_state_dict(torch.load("./checkpoints/aligner_prompting.pth", map_location=device))
        aligner.eval()
        print("Successfully loaded trained aligner!")
    except FileNotFoundError:
        print("WARNING: Could not find trained aligner_prompting checkpoint. Using untrained weights.")

    print(f"\nModel A Input: '{test_sentence}'")
    with torch.no_grad():
        # 1. Model A analyzes the text and produces a thought
        thought_a = pipeline.get_model_a_thought([test_sentence])
        thought_a = thought_a.to(next(aligner.parameters()).dtype)
        
        # 2. Aligner translates the thought into a Soft Prompt for Model B
        soft_prompt = aligner(thought_a) # [1, 1, dim_b]
        
        # Cast soft_prompt to match Model B's dtype
        soft_prompt = soft_prompt.to(pipeline.model_b.dtype)

        # 3. Model B generates text starting ONLY from this soft prompt
        # We need to construct generation inputs manually since we bypass input_ids
        
        print("-" * 40)
        print("Model B generating text from Model A's thought...")
        
        # Generation loop
        generated_ids = []
        current_embeds = soft_prompt
        
        # We'll generate up to 20 tokens to see what Model B "understood"
        for _ in range(20):
            outputs = pipeline.model_b(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            generated_ids.append(next_token_id.item())
            
            # Stop if EOS token
            if next_token_id.item() == pipeline.tokenizer_b.eos_token_id:
                break
                
            # Get embedding for the new token and append it
            new_embed = pipeline.model_b.get_input_embeddings()(next_token_id.unsqueeze(0))
            current_embeds = torch.cat([current_embeds, new_embed], dim=1)

        decoded_text = pipeline.tokenizer_b.decode(generated_ids, skip_special_tokens=True)
        print(f"Model B Output: {decoded_text}")
        print("-" * 40)

if __name__ == "__main__":
    test_sentence = "The Apollo 11 mission was the first time humans landed on the moon."
    test_latent_prompting(test_sentence)
