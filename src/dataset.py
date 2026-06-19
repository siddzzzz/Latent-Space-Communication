import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import os

class LatentCommunicationPipeline:
    def __init__(self, model_a_name: str, model_b_name: str, cache_dir: str = "./local_models"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading models on {self.device}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        os.makedirs(cache_dir, exist_ok=True)

        print(f"Loading Model A (Analyzer): {model_a_name}")
        self.tokenizer_a = AutoTokenizer.from_pretrained(model_a_name, cache_dir=cache_dir)
        self.model_a = AutoModelForCausalLM.from_pretrained(
            model_a_name, 
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_dir
        )
        self.model_a.eval()

        print(f"Loading Model B (Writer): {model_b_name}")
        self.tokenizer_b = AutoTokenizer.from_pretrained(model_b_name, cache_dir=cache_dir)
        self.model_b = AutoModelForCausalLM.from_pretrained(
            model_b_name, 
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_dir
        )
        self.model_b.eval()

        if self.tokenizer_a.pad_token is None:
            self.tokenizer_a.pad_token = self.tokenizer_a.eos_token
        if self.tokenizer_b.pad_token is None:
            self.tokenizer_b.pad_token = self.tokenizer_b.eos_token

    def get_model_a_thought(self, texts: list[str]):
        """
        Passes text through Model A and extracts a single 'thought' vector
        via mean-pooling the final hidden state.
        """
        with torch.no_grad():
            inputs_a = self.tokenizer_a(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            outputs_a = self.model_a(**inputs_a, output_hidden_states=True)
            hidden_a = outputs_a.hidden_states[-1] 
            
            mask_a = inputs_a['attention_mask'].unsqueeze(-1).expand(hidden_a.size()).float()
            pooled_a = torch.sum(hidden_a * mask_a, 1) / torch.clamp(mask_a.sum(1), min=1e-9)

        return pooled_a

def get_training_data(dataset_name="wikitext", subset="wikitext-2-raw-v1", split="train", num_samples=5000):
    print(f"Loading dataset {dataset_name} ({subset})...")
    dataset = load_dataset(dataset_name, subset, split=split)
    texts = [t["text"].strip() for t in dataset if len(t["text"].strip()) > 50]
    return texts[:num_samples]
