# GPT-2 From Scratch 🚀

This repository contains a complete implementation of GPT-2 from the ground up, refactored into a clean, modular Python codebase. It includes a custom LLM architecture, training pipelines, a spam classification system, and a modern web interface built with Chainlit.

## ✨ Features

- **Modular LLM Architecture**: Separate modules for Attention mechanisms, GPT model structure, and data loading.
- **Pre-trained Weights**: Support for loading GPT-2 Small (124M) weights for text generation.
- **Spam Classifier**: A specialized fine-tuned classifier using the custom GPT architecture to identify spam messages.
- **Chainlit Web UI**: A beautiful, interactive chat interface to interact with the LLM in real-time.
- **Optimized Performance**: Support for CUDA (NVIDIA) and MPS (Apple Silicon) acceleration.

## 📁 Project Structure

```text
.
├── architecture.py        # Core GPT-2 Model & Transformer layers
├── attention.py           # Multi-Head, Causal & Self-Attention modules
├── data.py                # Dataset loaders & tokenization wrappers
├── generate.py            # Text generation logic (top-k, temperature)
├── train.py               # Training & evaluation loops
├── app.py                 # Chainlit Web UI application
├── spam_classifier/       # Dedicated folder for spam detection
│   ├── spam_dataset.py    # Data processing for spam classification
│   ├── classifier.py      # Classification inference logic
│   └── test_spam.py       # Script to verify classifier performance
├── resources/             # Large model weights, data (ignored by git)
│   ├── gpt2-small-124M.pth
│   └── review_classifier.pth
└── notebooks_backup/      # Original research notebooks
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Git LFS](https://git-lfs.github.com/) (recommended for managing weights)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/llm-from-scratch.git
   cd llm-from-scratch
   ```

2. Install dependencies:
   ```bash
   pip install -r resources/requirements.txt
   pip install chainlit tiktoken torch pandas
   ```

3. Ensure your weights are in the `resources/` directory.

## 🎮 Usage

### Launching the Web UI
Interact with the model through a modern chat interface:
```bash
chainlit run app.py
```

### Testing the Spam Classifier
The spam classifier identifies whether a text is spam or not. To verify its performance:
```bash
python3 spam_classifier/test_spam.py
```

## 🛠 Refactoring Rules Applied

The codebase follows strict refactoring rules to ensure clarity and professional standards:
- **Clean Modularity**: Functions and classes are logically grouped into `.py` files.
- **Scratch vs. Library**: Scratch implementations of components (like LayerNorm or Tokenizers) are retained in comments for educational value, while optimized library functions are used for performance.
- **Separation of Concerns**: Specialized applications (like the Spam Classifier) are decoupled from the core LLM architecture.
- **GitHub Ready**: A robust `.gitignore` ensures that heavy training data and weights do not clutter your repository.

---
*Created as part of the "LLMs from Scratch" exploration.*
