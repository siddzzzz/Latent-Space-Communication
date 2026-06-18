import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import os

class HiddenStateDataset:
    def __init__(self, model_a_name: str, model_b_name: str, cache_dir: str = "./local_models"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading models on {self.device}...")
        
        # 4-bit quantization config to fit both models in 4GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        os.makedirs(cache_dir, exist_ok=True)

        print(f"Loading Model A: {model_a_name}")
        self.tokenizer_a = AutoTokenizer.from_pretrained(model_a_name, cache_dir=cache_dir)
        self.model_a = AutoModelForCausalLM.from_pretrained(
            model_a_name, 
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_dir
        )
        self.model_a.eval()

        print(f"Loading Model B: {model_b_name}")
        self.tokenizer_b = AutoTokenizer.from_pretrained(model_b_name, cache_dir=cache_dir)
        self.model_b = AutoModelForCausalLM.from_pretrained(
            model_b_name, 
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_dir
        )
        self.model_b.eval()

        # Fix missing pad tokens
        if self.tokenizer_a.pad_token is None:
            self.tokenizer_a.pad_token = self.tokenizer_a.eos_token
        if self.tokenizer_b.pad_token is None:
            self.tokenizer_b.pad_token = self.tokenizer_b.eos_token

    def get_hidden_states(self, texts: list[str]):
        """
        Takes a list of strings, runs them through both models,
        and returns the final layer hidden states.
        """
        # Ensure we don't track gradients for the frozen LLMs
        with torch.no_grad():
            # Model A
            inputs_a = self.tokenizer_a(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            outputs_a = self.model_a(**inputs_a, output_hidden_states=True)
            # The last hidden state before the LM head
            hidden_a = outputs_a.hidden_states[-1] 

            # Model B
            inputs_b = self.tokenizer_b(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            outputs_b = self.model_b(**inputs_b, output_hidden_states=True)
            # The last hidden state before the LM head
            hidden_b = outputs_b.hidden_states[-1] 
            
            # Note: Because Tokenizer A and Tokenizer B will tokenize the same text into DIFFERENT sequence lengths,
            # we cannot directly map [seq_len_A] to [seq_len_B] token-by-token.
            # To map concepts, we will mean-pool the sequence dimension to get a single vector per sentence.
            
            # Create attention masks for proper mean pooling
            mask_a = inputs_a['attention_mask'].unsqueeze(-1).expand(hidden_a.size()).float()
            pooled_a = torch.sum(hidden_a * mask_a, 1) / torch.clamp(mask_a.sum(1), min=1e-9)

            mask_b = inputs_b['attention_mask'].unsqueeze(-1).expand(hidden_b.size()).float()
            pooled_b = torch.sum(hidden_b * mask_b, 1) / torch.clamp(mask_b.sum(1), min=1e-9)

        return pooled_a, pooled_b

def get_training_data(dataset_name="wikitext", subset="wikitext-2-raw-v1", split="train", num_samples=1000):
    print(f"Loading dataset {dataset_name} ({subset})...")
    dataset = load_dataset(dataset_name, subset, split=split)
    
    # Filter out empty lines
    texts = [t["text"] for t in dataset if len(t["text"].strip()) > 20]
    return texts[:num_samples]
