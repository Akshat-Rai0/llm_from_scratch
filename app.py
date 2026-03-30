import asyncio
import chainlit as cl
import torch
import tiktoken
import os
from src.architecture import GPTModel
from src.generate import generate, text_to_token_ids, token_ids_to_text

# Configuration for GPT-2 Small
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

@cl.on_chat_start
async def start():
    cl.user_session.set("model_loaded", False)
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")

    cl.user_session.set("device", device)

    m = cl.Message(content="Loading model...")
    await m.send()

    try:
        model = GPTModel(GPT_CONFIG_124M)
        model.to(device)
        model.eval()

        if os.path.exists("resources/gpt2-small-124M.pth"):
            model.load_state_dict(torch.load("resources/gpt2-small-124M.pth", map_location=device, weights_only=True))
            m.content = "Model loaded successfully from GPT-2 weights! You can now send prompts."
        else:
            m.content = "No local model weights found. Using untrained model for demonstration."
            
        cl.user_session.set("model", model)
        cl.user_session.set("tokenizer", tiktoken.get_encoding("gpt2"))
        cl.user_session.set("model_loaded", True)
    except Exception as e:
        m.content = f"Failed to load model: {e}"
        
    await m.update()

@cl.on_message
async def main(message: cl.Message):
    if not cl.user_session.get("model_loaded"):
        await cl.Message(content="Model is not loaded properly. Please wait or check logs.").send()
        return

    model = cl.user_session.get("model")
    tokenizer = cl.user_session.get("tokenizer")
    device = cl.user_session.get("device")

    encoded = text_to_token_ids(message.content, tokenizer).to(device)
    context_size = GPT_CONFIG_124M["context_length"]

    msg = cl.Message(content="")
    await msg.send()

    try:
        # Offload the CPU/GPU-bound generate() call to a thread pool
        # so it does not block the async event loop
        token_ids = await asyncio.to_thread(
            generate,
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            top_k=25,
            temperature=1.4
        )
        
        # Decoding generated text
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        
        # The prompt is included in the decoded text for this basic generation
        msg.content = decoded_text
        await msg.update()

    except Exception as e:
        msg.content = f"Error generating text: {e}"
        await msg.update()
