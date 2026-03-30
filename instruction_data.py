
Copy

"""
instruction_data.py
-------------------
Data utilities for Instruction Finetuning (Chapter 7).
 
Provides:
  - download_and_load_file           : fetch/cache instruction JSON dataset
  - format_input                     : Alpaca-style prompt formatter
  - InstructionDataset               : PyTorch Dataset that pre-tokenises entries
  - custom_collate_fn                : collate with padding + ignore-index masking
  - create_instruction_dataloaders   : convenience factory for all 3 split loaders
"""
 
import json
import os
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
 
 
# ---------------------------------------------------------------------------
# Data download / load
# ---------------------------------------------------------------------------
 
def download_and_load_file(file_path: str, url: str) -> list:
    """Download *url* to *file_path* if it does not already exist,
    then parse and return the JSON list."""
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
 
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    return data
 
 
# ---------------------------------------------------------------------------
# Alpaca-style prompt formatter
# ---------------------------------------------------------------------------
 
def format_input(entry: dict) -> str:
    """Build the instruction prompt for a single dataset entry.
 
    Each entry is a dict with keys 'instruction', 'input', and 'output'.
    Returns the instruction + optional input portion (without the response).
    """
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
 
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text
 
 
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
 
class InstructionDataset(Dataset):
    """Pre-tokenises all entries at construction time.
 
    Args:
        data      : list of dicts with keys 'instruction', 'input', 'output'.
        tokenizer : a tiktoken (or compatible) tokenizer instance.
    """
 
    def __init__(self, data: list, tokenizer):
        self.data = data
        self.encoded_texts = []
 
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))
 
    def __getitem__(self, index: int):
        return self.encoded_texts[index]
 
    def __len__(self):
        return len(self.data)
 
 
# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------
 
def custom_collate_fn(
    batch,
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length=None,
    device: str = "cpu",
):
    """Pad a variable-length batch of token-id lists to uniform length.
 
    Targets are the inputs shifted one position to the right.
    All *additional* padding positions in the target (beyond the first one
    per sequence) are replaced with *ignore_index* so that cross-entropy
    does not penalise them.
 
    Optionally truncates every sequence to *allowed_max_length*.
 
    Args:
        batch              : list of token-id lists (Python ints).
        pad_token_id       : token id used for padding (GPT-2 EOS = 50256).
        ignore_index       : value used to mask padding in targets (-100).
        allowed_max_length : optional maximum sequence length (truncates if set).
        device             : torch device string for the returned tensors.
    """
    # Find the longest sequence in the batch and pad everything to that length
    # (+1 so we can create input/target pairs by shifting)
    batch_max_length = max(len(item) for item in batch) + 1
 
    inputs_lst, targets_lst = [], []
 
    for item in batch:
        new_item = item.copy()
        # Pad to batch_max_length
        new_item += [pad_token_id] * (batch_max_length - len(new_item))
 
        inputs = torch.tensor(new_item[:-1])   # drop last  → input sequence
        targets = torch.tensor(new_item[1:])   # shift right → target sequence
 
        # Mask all padding tokens in targets except the very first one so
        # the model learns to emit EOS but is not penalised for the rest.
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
 
        # Optional length cap
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
 
        inputs_lst.append(inputs)
        targets_lst.append(targets)
 
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
 
 
# ---------------------------------------------------------------------------
# Convenience dataloader factory
# ---------------------------------------------------------------------------
 
def create_instruction_dataloaders(
    data: list,
    tokenizer,
    device: str = "cpu",
    batch_size: int = 8,
    num_workers: int = 0,
    allowed_max_length: int = 1024,
    train_ratio: float = 0.85,
    test_ratio: float = 0.10,
    seed: int = 123,
):
    """Split *data* into train / test / validation sets and return three
    DataLoaders ready for instruction fine-tuning.
 
    Split sizes (default):
        train : 85 %
        test  : 10 %
        val   : 5 %  (remainder)
 
    Returns:
        train_loader, val_loader, test_loader, train_data, val_data, test_data
    """
    train_portion = int(len(data) * train_ratio)
    test_portion  = int(len(data) * test_ratio)
 
    train_data = data[:train_portion]
    test_data  = data[train_portion : train_portion + test_portion]
    val_data   = data[train_portion + test_portion :]
 
    # Build a partial collate function bound to the run-time device
    collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=allowed_max_length,
    )
 
    torch.manual_seed(seed)
 
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset   = InstructionDataset(val_data,   tokenizer)
    test_dataset  = InstructionDataset(test_data,  tokenizer)
 
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
 
    return train_loader, val_loader, test_loader, train_data, val_data, test_data
    