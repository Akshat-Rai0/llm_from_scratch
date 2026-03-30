# 🧠 LLM From Scratch

<div align="center">

**A complete, modular implementation of GPT-2 built from first principles in PyTorch.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Chainlit](https://img.shields.io/badge/Chainlit-UI-6B46C1?logo=chainlit&logoColor=white)](https://chainlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Based on the book [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka.*

</div>

---

## 📖 Overview

This repository implements the entire GPT-2 architecture and training pipeline from scratch, refactored from Jupyter notebooks into a clean, production-ready Python codebase. It covers every stage of the LLM lifecycle:

- **Tokenization & Data Loading** — BPE tokenization via `tiktoken`, sliding-window datasets
- **Core Architecture** — Multi-head causal self-attention, transformer blocks, layer norm
- **Pre-training** — Language model training loop with loss/perplexity tracking
- **Fine-tuning** — Spam classification (classification head) & instruction following (SFT)
- **Inference** — Top-k sampling, temperature scaling, greedy decoding
- **Interactive UI** — Real-time chat with the model via a Chainlit web interface

---

## 📁 Project Structure

```
llm_from_scratch/
│
├── app.py                      # 🚀 Entry point — Chainlit web UI
├── instruction_train.py        # CLI script for instruction fine-tuning (Ch. 7)
├── train.py                    # Training utilities (re-exported from src/)
│
├── src/                        # 📦 Core library package
│   ├── __init__.py
│   ├── attention.py            # SelfAttention, CausalAttention, MultiHeadAttention
│   ├── architecture.py         # LayerNorm, FeedForward, TransformerBlock, GPTModel
│   ├── data.py                 # GPTDatasetV1, create_dataloader_v1
│   ├── generate.py             # generate(), text_to_token_ids(), token_ids_to_text()
│   ├── train.py                # calc_loss_batch/loader, evaluate_model, train_model_simple
│   ├── instruction_data.py     # InstructionDataset, custom_collate_fn, dataloaders
│   └── instruction_train.py    # Weight loading, fine-tuning logic, CLI entrypoint
│
├── spam_classifier/            # 📦 Spam detection sub-package
│   ├── __init__.py
│   ├── spam_dataset.py         # SpamDataset and dataloader factory
│   ├── classifier.py           # calc_accuracy_loader(), classify_review()
│   └── test_spam.py            # Inference script to verify classifier
│
├── resources/                  # ⚠️  Not committed to Git (large files)
│   ├── gpt2-small-124M.pth     # GPT-2 Small pre-trained weights
│   ├── gpt2-medium355M-sft.pth # Instruction fine-tuned weights
│   ├── review_classifier.pth   # Spam classifier weights
│   ├── requirements.txt        # Python dependencies
│   └── ...                     # Training data, plots, etc.
│
├── notebooks_backup/           # 📓 Original research notebooks (Ch. 1–7)
│   ├── WORKING WITH DATA(CHAPTER 1).ipynb
│   ├── ATTENTION MECHANISM(CHAPTER 2).ipynb
│   ├── IMPLEMENTING LLM ARCHITECHTURE(CHAPTER 3).ipynb
│   ├── PRETRAINING ON UNLABELED DATA (CHAPTER 4).ipynb
│   ├── FINETUNING FOR CLASSIFICATION(CHAPTER 5).ipynb
│   ├── INSTRUCTION FINETUNING (6).ipynb
│   ├── CHAPTER 6.ipynb
│   └── gpt_download.py         # Helper to download OpenAI GPT-2 checkpoints
│
├── chainlit.md                 # Chainlit welcome message
├── .gitignore
└── README.md
```

---

## 🏗️ Architecture

```
Input Tokens
     │
     ▼
┌─────────────────────────────┐
│   Token Embedding (50257)   │
│ + Positional Embedding      │
│ + Dropout                   │
└────────────┬────────────────┘
             │
     ┌───────▼────────┐  ×N layers
     │ TransformerBlock│
     │ ┌─────────────┐ │
     │ │  LayerNorm  │ │
     │ │      ↓      │ │
     │ │MultiHeadAttn│ │  (Causal, h=12, d=768)
     │ │      ↓      │ │
     │ │  + Residual │ │
     │ │  LayerNorm  │ │
     │ │      ↓      │ │
     │ │ FeedForward │ │  (4× expansion, GELU)
     │ │      ↓      │ │
     │ │  + Residual │ │
     └─┴─────────────┴─┘
             │
     ┌───────▼────────┐
     │  Final LayerNorm│
     └───────┬─────────┘
             │
     ┌───────▼────────┐
     │   Output Head  │  (vocab logits or classification)
     └────────────────┘
```

### Supported GPT-2 Variants

| Model            | Parameters | Layers | Heads | Embedding Dim |
|------------------|-----------|--------|-------|---------------|
| GPT-2 Small      | 124M       | 12     | 12    | 768           |
| GPT-2 Medium     | 355M       | 24     | 16    | 1024          |
| GPT-2 Large      | 774M       | 36     | 20    | 1280          |
| GPT-2 XL         | 1558M      | 48     | 25    | 1600          |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- macOS (MPS), Linux (CUDA), or any CPU system

### 1. Clone the Repository

```bash
git clone https://github.com/Akshat-Rai0/llm_from_scratch.git
cd llm_from_scratch
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r resources/requirements.txt
```

> **Note:** Model weight files (`.pth`) are not included in the repository. Place them in the `resources/` directory before running any scripts.

---

## 🎮 Usage

### 🌐 Launch the Chat UI

Interact with GPT-2 Small via a beautiful Chainlit interface:

```bash
chainlit run app.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser. The UI will automatically load weights from `resources/gpt2-small-124M.pth` if present.

---

### 📝 Instruction Fine-Tuning (Chapter 7 — SFT)

Fine-tune GPT-2 to follow natural language instructions using the Alpaca-format dataset:

```bash
# Fine-tune GPT-2 Medium (default)
python instruction_train.py

# Fine-tune GPT-2 Small for 2 epochs
python instruction_train.py --model "gpt2-small (124M)" --epochs 2

# All available options
python instruction_train.py --help
```

**CLI Arguments:**

| Argument              | Default                  | Description                              |
|-----------------------|--------------------------|------------------------------------------|
| `--model`             | `gpt2-medium (355M)`     | GPT-2 variant to fine-tune               |
| `--epochs`            | `1`                      | Number of training epochs                |
| `--batch-size`        | `2`                      | Training batch size                      |
| `--lr`                | `5e-5`                   | Learning rate (AdamW)                    |
| `--weight-decay`      | `0.1`                    | Weight decay (AdamW)                     |
| `--allowed-max-length`| `512`                    | Max token length per sequence            |
| `--data-file`         | `instruction-data.json`  | Path to local instruction dataset        |
| `--response-file`     | `...-with-response.json` | Output path for test set predictions     |

---

### 🚫 Spam Classifier

Test the fine-tuned binary spam classifier:

```bash
python spam_classifier/test_spam.py
```

**Example output:**
```
--- Spam Classifier Test ---

Text: "You are a winner! You have been specially selected..."
Prediction: SPAM

Text: "Hey, just wanted to check if we're still on for dinner tonight?"
Prediction: NOT SPAM
```

---

## 🧩 Module Reference

### `src/attention.py`

| Class / Function     | Description                                           |
|----------------------|-------------------------------------------------------|
| `SelfAttention_v2`   | Basic self-attention (educational reference)          |
| `CausalAttention`    | Single-head causal attention with dropout & masking   |
| `MultiHeadAttention` | Production multi-head causal attention (used in GPT)  |

### `src/architecture.py`

| Class / Function  | Description                                              |
|-------------------|----------------------------------------------------------|
| `LayerNorm`       | Custom layer normalization (from scratch)                |
| `FeedForward`     | Two-layer MLP with GELU activation (4× expansion)        |
| `TransformerBlock`| Pre-LN transformer block with residual connections       |
| `GPTModel`        | Full GPT-2 decoder-only model                            |

### `src/generate.py`

| Function               | Description                                           |
|------------------------|-------------------------------------------------------|
| `generate()`           | Autoregressive generation (top-k + temperature)       |
| `text_to_token_ids()`  | Encode text string to token tensor                    |
| `token_ids_to_text()`  | Decode token tensor back to string                    |

### `src/train.py`

| Function                 | Description                                         |
|--------------------------|-----------------------------------------------------|
| `calc_loss_batch()`      | Cross-entropy loss for a single batch               |
| `calc_loss_loader()`     | Average loss over a DataLoader                      |
| `evaluate_model()`       | Compute train/val loss in eval mode                 |
| `train_model_simple()`   | Full training loop with logging and sampling        |

### `src/instruction_data.py`

| Class / Function                  | Description                                     |
|-----------------------------------|-------------------------------------------------|
| `download_and_load_file()`        | Fetch & cache instruction JSON dataset          |
| `format_input()`                  | Alpaca-style prompt formatter                   |
| `InstructionDataset`              | PyTorch Dataset with pre-tokenized entries      |
| `custom_collate_fn()`             | Padding + ignore-index masking for SFT batches  |
| `create_instruction_dataloaders()`| Factory for train/val/test DataLoaders          |

### `spam_classifier/`

| File              | Description                                                |
|-------------------|------------------------------------------------------------|
| `spam_dataset.py` | `SpamDataset` and dataloader factory for SMS Spam corpus   |
| `classifier.py`   | `calc_accuracy_loader()`, `classify_review()` inference    |
| `test_spam.py`    | End-to-end test: load weights → classify sample texts      |

---

## ⚙️ Hardware Acceleration

The codebase auto-detects the best available device:

| Platform         | Backend       | Notes                                          |
|------------------|---------------|------------------------------------------------|
| NVIDIA GPU       | `cuda`        | Fastest; recommended for training              |
| Apple Silicon    | `mps`         | Requires PyTorch ≥ 2.9 for MPS stability       |
| CPU              | `cpu`         | Falls back automatically                       |

---

## 📚 Chapter Mapping

| Chapter | Topic                              | Notebooks                                    | Source Module(s)                          |
|---------|------------------------------------|----------------------------------------------|-------------------------------------------|
| Ch. 1   | Working with Text Data             | `WORKING WITH DATA(CHAPTER 1).ipynb`         | `src/data.py`                             |
| Ch. 2   | Attention Mechanisms               | `ATTENTION MECHANISM(CHAPTER 2).ipynb`       | `src/attention.py`                        |
| Ch. 3   | GPT-2 Architecture                 | `IMPLEMENTING LLM ARCHITECHTURE(CHAPTER 3).ipynb` | `src/architecture.py`                |
| Ch. 4   | Pre-training on Unlabeled Data     | `PRETRAINING ON UNLABELED DATA (CHAPTER 4).ipynb` | `src/train.py`, `src/generate.py`    |
| Ch. 5   | Fine-tuning for Classification     | `FINETUNING FOR CLASSIFICATION(CHAPTER 5).ipynb` | `spam_classifier/`                   |
| Ch. 6   | Loading Pre-trained Weights        | `CHAPTER 6.ipynb`                            | `app.py`, `src/architecture.py`           |
| Ch. 7   | Instruction Fine-tuning (SFT)      | `INSTRUCTION FINETUNING (6).ipynb`           | `src/instruction_train.py`, `src/instruction_data.py` |

---

## 🛠️ Design Principles

- **Educational Integrity** — Scratch implementations (e.g., `LayerNorm`, `SelfAttention`) are retained and explained, while optimized library equivalents are used where appropriate.
- **Separation of Concerns** — Core model (`src/`), spam classifier (`spam_classifier/`), and UI (`app.py`) are fully decoupled.
- **Clean CLI** — `instruction_train.py` is runnable from the command line with sane defaults and `--help` documentation.
- **Async UI** — The Chainlit app offloads model inference to a thread pool to keep the event loop non-blocking.
- **Git-Friendly** — `.gitignore` excludes all large model weights (`.pth`), training data, and environment files.

---

## 📦 Dependencies

See [`resources/requirements.txt`](resources/requirements.txt) for the full list. Key packages:

```
torch>=2.0
tiktoken
chainlit
numpy
tqdm
requests
pandas
```

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

*Built while reading [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) — a hands-on journey from tokens to transformer.*

</div>
