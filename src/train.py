import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch.nn.functional as F

from dataset import LatentCommunicationPipeline, get_training_data
from models.bottleneck import LatentAligner

MODEL_A = "Qwen/Qwen2.5-0.5B"
MODEL_B = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 10
LEARNING_RATE = 5e-4
NUM_SAMPLES = 5000

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training using device: {device}")

    pipeline = LatentCommunicationPipeline(model_a_name=MODEL_A, model_b_name=MODEL_B)
    texts = get_training_data(num_samples=NUM_SAMPLES)
    dataloader = DataLoader(texts, batch_size=BATCH_SIZE, shuffle=True)

    dim_a = pipeline.model_a.config.hidden_size
    dim_b = pipeline.model_b.config.hidden_size
    
    aligner = LatentAligner(input_dim=dim_a, output_dim=dim_b).to(device)
    optimizer = torch.optim.AdamW(aligner.parameters(), lr=LEARNING_RATE)

    aligner.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch_texts in enumerate(progress_bar):
            
            # 1. Get Model A's final thought (frozen)
            thought_a = pipeline.get_model_a_thought(batch_texts)

            # Cast thought_a to match aligner's dtype (float32)
            thought_a = thought_a.to(next(aligner.parameters()).dtype)

            # 2. Translate thought into a Soft Prompt for Model B
            # shape: [batch, 1, dim_b]
            soft_prompt = aligner(thought_a)

            # 3. Get Model B's standard token embeddings for the target text
            inputs_b = pipeline.tokenizer_b(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            input_ids_b = inputs_b['input_ids']
            
            with torch.no_grad():
                text_embeds_b = pipeline.model_b.get_input_embeddings()(input_ids_b)
            
            # Cast soft_prompt to match Model B's embeddings dtype (float16)
            soft_prompt = soft_prompt.to(text_embeds_b.dtype)

            # 4. Concatenate: [Soft_Prompt, Text_Embeddings]
            # This makes Model B read the prompt, then the text
            # shape: [batch, 1 + seq_len, dim_b]
            full_embeds = torch.cat([soft_prompt, text_embeds_b], dim=1)
            
            # We must also pad the attention mask to account for the +1 length
            mask_b = inputs_b['attention_mask']
            prompt_mask = torch.ones((mask_b.size(0), 1), device=device)
            full_mask = torch.cat([prompt_mask, mask_b], dim=1)

            # 5. Run Model B (Forward pass only, backprop happens through inputs_embeds down to aligner)
            outputs = pipeline.model_b(inputs_embeds=full_embeds, attention_mask=full_mask)
            logits = outputs.logits

            # 6. Calculate Cross Entropy Loss
            # We want the soft prompt to predict the first token, the first token to predict the second, etc.
            # Shift logits and labels. 
            # Logits shape: [batch, 1+seq_len, vocab_size]
            # Labels shape: [batch, seq_len]
            # We shift logits by discarding the last prediction, and labels remain as is
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids_b.contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1),
                ignore_index=pipeline.tokenizer_b.pad_token_id
            )
            
            # Scale loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(aligner.state_dict(), "./checkpoints/aligner_prompting.pth")
    print("Training complete! Aligner saved to ./checkpoints/aligner_prompting.pth")

if __name__ == "__main__":
    train()
