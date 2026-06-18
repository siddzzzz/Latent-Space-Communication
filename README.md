# Shared Latent Workspace: Beyond Natural Language Communication Between Specialized LLMs

This repository explores a foundational question in multi-agent systems:
**Can specialized language models communicate directly through learned latent representations instead of natural language?**

By exchanging continuous vector representations rather than discrete natural language tokens, we bypass the textual bottleneck. This allows specialized models to exchange rich semantic state directly, saving inference cost and avoiding representation loss.

---

## 🚀 Key Features

* **Cross-Architecture Communication**: Bridge models with completely different structures, tokenizers, and hidden dimensions (e.g., Llama and Qwen) via continuous representations.
* **Training-Free SVD/Procrustes Alignment**: Align different embedding spaces mathematically in milliseconds using Singular Value Decomposition on shared concept anchors.
* **Parameter-Efficient Adapter Training**: Train a tiny connector module while keeping both LLMs completely frozen. This enables learning the communication protocol without altering base model capabilities.
* **4GB VRAM Optimization**: Fully executable on consumer-grade hardware (such as an NVIDIA RTX 3050 4GB) using FP16 mixed precision, frozen model states, and gradient accumulation.
* **Automatic Model Cache**: Downloads and caches Hugging Face model weight files directly inside the workspace directory (`./local_models/`) to keep your host filesystem clean.

---

## 📐 Conceptual Architecture

Normally, multi-agent systems use text prompts:
```text
Codebase ➔ Model A (Code Expert) ➔ Text Summary ("This code prints...") ➔ Model B (Report Expert) ➔ Final Report
```
The **Shared Latent Workspace** replaces the text bottleneck with a continuous translation adapter:
```text
Input Context (Code)
       ↓
Model A (Qwen-Coder-0.5B - Frozen)
       ↓  (Extract last-layer hidden states)
Latent Projector (Perceiver / MLP Adapter - Trainable)
       ↓  (Fixed-size latent sequence of length K)
Model B (Qwen-Instruct-0.5B - Frozen)
       ↓  (Injected as Soft-Prompt Prefix)
Final Report Generation
```

### What is a Soft Prompt Prefix?
Unlike hard prompts (text characters that go through tokenization and embedding lookup), a **soft prompt** consists of continuous vectors injected directly into the LLM's attention layers. They represent virtual tokens carrying multi-dimensional semantic concepts directly from the sender model's latent workspace.

---

## 📁 Repository Structure

```text
├── src/
│   ├── models/
│   │   └── bottleneck.py  # LatentConnector & pipeline wrapper
│   ├── aligner.py         # Closed-form SVD Procrustes alignment calculations
│   ├── dataset.py         # Programmatic data generator & tokenizer utilities
│   ├── train.py           # VRAM-optimized training loop
│   ├── run_demo.py        # Pipeline demonstrator & baseline comparator
│   └── test_pipeline.py   # Dry-run unit tests utilizing mock tensors
├── requirements.txt       # Project python dependencies
└── README.md              # Documentation
```

---

## 🛠️ Step-by-Step Setup & Execution

### 1. Environment Installation
Ensure you have a Conda environment activated with PyTorch and CUDA support.

```bash
# Clone the repository
git clone <your-repo-link>
cd Latent-Space-Communication

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Your Architecture (Dry Run)
Before downloading any model weights, check the pipeline and tensor geometry using our dry-run test suite. This runs with mock tensors and finishes instantly.

```bash
python src/test_pipeline.py
```

### 3. Option A: Run Zero-Shot SVD Mapping (Training-Free)
If you want to run the pipeline instantly without waiting for training:
The demo script will automatically calculate a closed-form Procrustes projection matrix across $2,000$ alphanumeric words shared by both models' vocabularies.

```bash
python src/run_demo.py
```

### 4. Option B: Train the Latent Adapter
To let the adapter learn a custom translator between the two specialized models, run the training script. This will use a local, offline synthetic dataset of 300 code-to-summary pairs and train the connector in a few minutes.

```bash
python src/train.py --epochs 3 --batch_size 1 --grad_accum 8 --lr 2e-4
```

*Note: Model checkpoints are saved locally under `./checkpoints/connector_best.pt` (size is only ~25MB).*

### 5. Run the Comparative Evaluation
After training, run the demo script to compare performance:
1. **Method 1 (Latent Workspace)**: Model A $\rightarrow$ Trained Latent Connector $\rightarrow$ Model B.
2. **Method 2 (Text-Based Summary)**: Model A $\rightarrow$ Generate Text Summary $\rightarrow$ Model B.
3. **Method 3 (Control)**: Model B generating without any context.

```bash
python src/run_demo.py
```

---

## 🎓 Mathematical Formulation of Procrustes Alignment

For two different embedding spaces representing a vocabulary matrix $X_A \in \mathbb{R}^{N \times d_A}$ and $X_B \in \mathbb{R}^{N \times d_B}$:

1. **Mean Centering**:
   $$\bar{X}_A = X_A - \mu_A, \quad \bar{X}_B = X_B - \mu_B$$
2. **Covariance matrix**:
   $$C = \bar{X}_A^T \bar{X}_B$$
3. **Singular Value Decomposition**:
   $$U, \Sigma, V^T = \text{SVD}(C)$$
4. **Optimal Rotation Matrix**:
   $$W = U V^T$$
5. **Latent Space Translation**:
   $$\mathbf{h}_B = (\mathbf{h}_A - \mu_A) W + \mu_B$$
   
This alignment transforms activations from Model A's coordinate space directly into Model B's coordinates.