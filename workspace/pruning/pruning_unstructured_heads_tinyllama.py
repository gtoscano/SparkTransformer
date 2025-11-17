# pruning_unstructured_heads_tinyllama.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from latency_bench import bench_generation
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if device.type == "cuda" else torch.float32,
    #torch_dtype=torch.float32,
).to(device)


model.config.eos_token_id = tok.eos_token_id
model.config.pad_token_id = tok.pad_token_id

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model: {model_name}, params: {n_params:.1f}M, dtype: {next(model.parameters()).dtype}, device: {next(model.parameters()).device}")

print("Baseline:", bench_generation(model, tok, "Why do we prune attention heads?"))

# ---- Unstructured head pruning: zero some heads but keep shape ----
def zero_out_heads(model, heads_to_zero):
    """
    heads_to_zero: dict[layer_idx] = [head_indices]
    """
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in heads_to_zero:
            continue
        hlist = heads_to_zero[layer_idx]
        attn = layer.self_attn

        for h in hlist:
            start = h * head_dim
            end = (h + 1) * head_dim
            # q, k, v projections
            attn.q_proj.weight.data[start:end, :] = 0
            attn.k_proj.weight.data[start:end, :] = 0
            attn.v_proj.weight.data[start:end, :] = 0
            if hasattr(attn, "o_proj"):
                attn.o_proj.weight.data[:, start:end] = 0

heads_to_zero = {i: [0, 1] for i in range(4)}  # zero 2 heads in first 4 layers
zero_out_heads(model, heads_to_zero)
print("Zeroed heads (unstructured at head level).")

print("After zeroing heads:", bench_generation(model, tok, "Why do we prune attention heads?"))

