"""
instruction_train.py
--------------------
End-to-end instruction fine-tuning script (Chapter 7).

Usage
-----
    python instruction_train.py                     # defaults
    python instruction_train.py --model gpt2-small (124M) --epochs 2

What it does
------------
1. Downloads / loads the instruction dataset (instruction-data.json).
2. Splits into train / val / test sets and builds DataLoaders.
3. Downloads GPT-2 pre-trained weights and loads them into GPTModel.
4. Fine-tunes with AdamW using train_model_simple.
5. Generates responses for the first 3 test entries (qualitative check).
6. Runs the full test set and saves results to instruction-data-with-response.json.
7. Saves the fine-tuned weights (e.g. gpt2-medium355M-sft.pth).
"""

import os
import re
import json
import time
import argparse

import torch
import tiktoken
from tqdm import tqdm

# ------------------------------------------------------------------
# Local modules (same package)
# ------------------------------------------------------------------
from instruction_data import (
    download_and_load_file,
    format_input,
    create_instruction_dataloaders,
)
from architecture import GPTModel
from generate import generate, text_to_token_ids, token_ids_to_text
from train import calc_loss_loader, train_model_simple


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0,       # Dropout rate (0 for fine-tuning)
    "qkv_bias": True,       # Query-key-value bias
}

MODEL_CONFIGS = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


# ------------------------------------------------------------------
# Device selection
# ------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available torch device (CUDA > MPS >= 2.9 > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------
# Weight loading (uses TF checkpoint via gpt_download helper)
# ------------------------------------------------------------------

def load_pretrained_gpt2(model_name: str, models_dir: str = "gpt2") -> GPTModel:
    """Download GPT-2 weights (TF checkpoint) and load them into GPTModel.

    Relies on the gpt_download.py helper already present in the repo
    (notebooks_backup/gpt_download.py or local gpt_download.py).
    """
    # Allow importing from notebooks_backup if no local copy exists
    import sys
    if not os.path.exists("gpt_download.py"):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "notebooks_backup"))

    from gpt_download import download_and_load_gpt2

    cfg = {**BASE_CONFIG, **MODEL_CONFIGS[model_name]}
    model_size = model_name.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir=models_dir)

    model = GPTModel(cfg)
    _load_weights_into_gpt(model, params)
    return model, cfg


def _load_weights_into_gpt(gpt: GPTModel, params: dict):
    """Copy OpenAI TF checkpoint weights into our GPTModel.

    This mirrors the load_weights_into_gpt function from previous_chapters.py
    but operates directly on GPTModel defined in architecture.py.
    """
    gpt.pos_emb.weight = torch.nn.Parameter(torch.tensor(params["wpe"]))
    gpt.tok_emb.weight = torch.nn.Parameter(torch.tensor(params["wte"]))

    for b in range(len(gpt.trf_blocks)):
        q, k, v = torch.split(
            torch.tensor(params["blocks"][b]["attn"]["c_attn"]["w"]).T,
            gpt.trf_blocks[b].att.W_query.weight.shape[0],
            dim=0,
        )
        gpt.trf_blocks[b].att.W_query.weight = torch.nn.Parameter(q)
        gpt.trf_blocks[b].att.W_key.weight   = torch.nn.Parameter(k)
        gpt.trf_blocks[b].att.W_value.weight = torch.nn.Parameter(v)

        q_bias, k_bias, v_bias = torch.split(
            torch.tensor(params["blocks"][b]["attn"]["c_attn"]["b"]),
            gpt.trf_blocks[b].att.W_query.bias.shape[0],
        )
        gpt.trf_blocks[b].att.W_query.bias = torch.nn.Parameter(q_bias)
        gpt.trf_blocks[b].att.W_key.bias   = torch.nn.Parameter(k_bias)
        gpt.trf_blocks[b].att.W_value.bias = torch.nn.Parameter(v_bias)

        gpt.trf_blocks[b].att.out_proj.weight = torch.nn.Parameter(
            torch.tensor(params["blocks"][b]["attn"]["c_proj"]["w"]).T)
        gpt.trf_blocks[b].att.out_proj.bias = torch.nn.Parameter(
            torch.tensor(params["blocks"][b]["attn"]["c_proj"]["b"]))

        gpt.trf_blocks[b].ff.layers[0].weight = torch.nn.Parameter(
            torch.tensor(params["blocks"][b]["mlp"]["c_fc"]["w"]).T)
        gpt.trf_blocks[b].ff.layers[0].bias = torch.nn.Parameter(
            torch.tensor(params["blocks"][b]["mlp"]["c_fc"]["b"]))
        gpt.trf_blocks[b].ff.layers[2].weight = torch.nn.Parameter(
            torch.tensor(params["blocks"][b]["mlp"]["c_proj"]["w"]).T)
        gpt.trf_blocks[b].ff.layers[2].bias = torch.nn.Parameter(
            torch.tensor(params["blocks"][b]["mlp"]["c_proj"]["b"]))

        gpt.trf_blocks[b].norm1.scale = torch.nn.Parameter(
            torch.tensor(params["blocks"][b]["ln_1"]["g"]))
        gpt.trf_blocks[b].norm1.shift = torch.nn.Parameter(
            torch.tensor(params["blocks"][b]["ln_1"]["b"]))
        gpt.trf_blocks[b].norm2.scale = torch.nn.Parameter(
            torch.tensor(params["blocks"][b]["ln_2"]["g"]))
        gpt.trf_blocks[b].norm2.shift = torch.nn.Parameter(
            torch.tensor(params["blocks"][b]["ln_2"]["b"]))

    gpt.final_norm.scale = torch.nn.Parameter(torch.tensor(params["g"]))
    gpt.final_norm.shift = torch.nn.Parameter(torch.tensor(params["b"]))
    gpt.out_head.weight   = torch.nn.Parameter(torch.tensor(params["wte"]))


# ------------------------------------------------------------------
# Evaluation helpers (qualitative + quantitative)
# ------------------------------------------------------------------

def generate_response(model, tokenizer, input_text: str, cfg: dict, device) -> str:
    """Run greedy generation and extract the response portion."""
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=cfg["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    return (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )


def evaluate_on_test_set(model, tokenizer, test_data: list, cfg: dict, device) -> list:
    """Generate a response for every entry in *test_data* and return the
    augmented list (adds 'model_response' key to each entry)."""
    for i, entry in tqdm(enumerate(test_data), total=len(test_data), desc="Test set"):
        input_text = format_input(entry)
        test_data[i]["model_response"] = generate_response(
            model, tokenizer, input_text, cfg, device
        )
    return test_data


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    # ----------------------------------------------------------
    # 0. Environment
    # ----------------------------------------------------------
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    device = get_device()
    print(f"Device: {device}")

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    data = download_and_load_file(args.data_file, DATA_URL)
    print(f"Number of entries: {len(data)}")

    tokenizer = tiktoken.get_encoding("gpt2")

    train_loader, val_loader, test_loader, train_data, val_data, test_data = (
        create_instruction_dataloaders(
            data,
            tokenizer,
            device=str(device),
            batch_size=args.batch_size,
            allowed_max_length=args.allowed_max_length,
        )
    )
    print(
        f"Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}"
    )

    # ----------------------------------------------------------
    # 2. Load pre-trained model
    # ----------------------------------------------------------
    print(f"\nLoading {args.model} weights …")
    model, cfg = load_pretrained_gpt2(args.model, models_dir=args.gpt2_dir)
    model.to(device)
    model.eval()

    # ----------------------------------------------------------
    # 3. Baseline loss (before fine-tuning)
    # ----------------------------------------------------------
    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=5)
    print(f"\nPre-training loss  — train: {train_loss:.4f}  val: {val_loss:.4f}")

    # Quick qualitative check before training
    print("\n--- Sample response BEFORE fine-tuning ---")
    sample_input = format_input(val_data[0])
    print(sample_input)
    print("\n>>> " + generate_response(model, tokenizer, sample_input, cfg, device))

    # ----------------------------------------------------------
    # 4. Fine-tune
    # ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"\nFine-tuning for {args.epochs} epoch(s) …")
    start = time.time()
    torch.manual_seed(123)

    train_losses, val_losses, _ = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
    )

    elapsed = (time.time() - start) / 60
    print(f"Training completed in {elapsed:.2f} minutes.")

    # ----------------------------------------------------------
    # 5. Qualitative evaluation on first 3 test entries
    # ----------------------------------------------------------
    print("\n--- Sample responses AFTER fine-tuning ---")
    torch.manual_seed(123)
    for entry in test_data[:3]:
        input_text = format_input(entry)
        response   = generate_response(model, tokenizer, input_text, cfg, device)
        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response}")
        print("-" * 40)

    # ----------------------------------------------------------
    # 6. Full test-set evaluation → JSON
    # ----------------------------------------------------------
    print("\nGenerating responses for the full test set …")
    test_data = evaluate_on_test_set(model, tokenizer, test_data, cfg, device)

    with open(args.response_file, "w") as f:
        json.dump(test_data, f, indent=4)
    print(f"Saved test responses to {args.response_file}")

    # ----------------------------------------------------------
    # 7. Save fine-tuned weights
    # ----------------------------------------------------------
    model_tag  = re.sub(r"[ ()]", "", args.model)
    save_path  = f"{model_tag}-sft.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instruction fine-tuning for GPT-2")

    parser.add_argument(
        "--model",
        default="gpt2-medium (355M)",
        choices=list(MODEL_CONFIGS.keys()),
        help="GPT-2 variant to fine-tune.",
    )
    parser.add_argument(
        "--data-file",
        default="instruction-data.json",
        help="Path to the local instruction dataset JSON file.",
    )
    parser.add_argument(
        "--gpt2-dir",
        default="gpt2",
        help="Directory for GPT-2 checkpoint downloads.",
    )
    parser.add_argument("--epochs",          type=int,   default=1)
    parser.add_argument("--batch-size",      type=int,   default=2)
    parser.add_argument("--allowed-max-length", type=int, default=512)
    parser.add_argument("--lr",              type=float, default=5e-5)
    parser.add_argument("--weight-decay",    type=float, default=0.1)
    parser.add_argument(
        "--response-file",
        default="instruction-data-with-response.json",
        help="Where to save the annotated test-set responses.",
    )

    main(parser.parse_args())
