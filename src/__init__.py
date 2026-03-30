"""
src — Core library for the LLM From Scratch project.

Submodules:
    attention        — Self-attention, causal attention, multi-head attention
    architecture     — LayerNorm, FeedForward, TransformerBlock, GPTModel
    data             — GPTDatasetV1, create_dataloader_v1
    generate         — generate(), text_to_token_ids(), token_ids_to_text()
    train            — Training loops, loss helpers, evaluation utilities
    instruction_data — Instruction fine-tuning dataset utilities
    instruction_train— End-to-end SFT script and weight loading
"""

from .architecture import GPTModel, LayerNorm, FeedForward, TransformerBlock
from .attention import MultiHeadAttention, CausalAttention, SelfAttention_v2
from .generate import generate, text_to_token_ids, token_ids_to_text
from .train import (
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    train_model_simple,
)
from .data import GPTDatasetV1, create_dataloader_v1

__all__ = [
    # Architecture
    "GPTModel",
    "LayerNorm",
    "FeedForward",
    "TransformerBlock",
    # Attention
    "MultiHeadAttention",
    "CausalAttention",
    "SelfAttention_v2",
    # Generation
    "generate",
    "text_to_token_ids",
    "token_ids_to_text",
    # Training
    "calc_loss_batch",
    "calc_loss_loader",
    "evaluate_model",
    "train_model_simple",
    # Data
    "GPTDatasetV1",
    "create_dataloader_v1",
]
