import torch
import tiktoken
import sys
import os

# Adjust path to import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture import GPTModel
from spam_classifier.classifier import classify_review

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        device = torch.device("mps") if (major, minor) >= (2, 9) else torch.device("cpu")
    else:
        device = torch.device("cpu")

    # The original spam classifier was trained with GPT2-small
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768, 
        "n_layers": 12, 
        "n_heads": 12
    }
    
    # Initialize the model
    model = GPTModel(BASE_CONFIG)
    
    # Modify the last layer for binary classification instead of vocab prediction
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=2)
    model.to(device)

    # Load weights
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(base_dir, "resources", "review_classifier.pth")
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    # Let's define some test samples
    test_samples = [
        "You are a winner! You have been specially selected to receive $1000 cash or a $2000 award.",
        "Hey, just wanted to check if we're still on for dinner tonight? Let me know!",
        "URGENT! Your mobile number has been awarded with a £2000 prize GUARANTEED.",
        "Did you finish reviewing the pull requests?"
    ]

    print("--- Spam Classifier Test ---")
    for text in test_samples:
        # Pass a reasonable max_length used during training, e.g., 120
        result = classify_review(text, model, tokenizer, device, max_length=120)
        print(f"\nText: \"{text}\"")
        print(f"Prediction: {result.upper()}")


if __name__ == "__main__":
    main()
