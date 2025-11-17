# pruning_structured_mlp_gpt2.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from latency_bench import bench_generation

model_name = "gpt2"

# ---- tokenizer ----
tok = AutoTokenizer.from_pretrained(model_name)

# GPT-2 has no pad token → map pad to EOS
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- model (float32) ----
# Load in float32, prune, then optionally cast to float16 on CUDA
model = AutoModelForCausalLM.from_pretrained(model_name)  # float32 by default
model.to(device)

model.config.eos_token_id = tok.eos_token_id
model.config.pad_token_id = tok.pad_token_id

def get_mlp_hidden_dim(m):
    # use first block as reference
    mlp = m.transformer.h[0].mlp
    # c_fc: Conv1D(intermediate_size, hidden_size) → weight shape (hidden_size, intermediate_size)
    return mlp.c_fc.weight.shape[1]

print(f"Original MLP hidden dim: {get_mlp_hidden_dim(model)}")

# ---- baseline BEFORE pruning ----
baseline = bench_generation(
    model,
    tok,
    prompt="Why should we prune MLP neurons?",
    iters=20,
    max_new_tokens=20,
)
print("Baseline:", baseline)


def prune_mlp_neurons(model, keep_ratio=0.7):
    """
    Structured MLP neuron pruning for GPT-2:
    - Each block.mlp has:
        c_fc:  Conv1D(intermediate_size, hidden_size)
        c_proj: Conv1D(hidden_size, intermediate_size)
    - We compute an importance score per hidden neuron (intermediate dim)
      and keep top `keep_ratio` of them, shrinking the MLP width.
    """
    for block_idx, block in enumerate(model.transformer.h):
        mlp = block.mlp
        c_fc = mlp.c_fc   # Conv1D(intermediate_size, hidden_size)
        c_proj = mlp.c_proj  # Conv1D(hidden_size, intermediate_size)

        # c_fc.weight: shape (hidden_size, intermediate_size)
        W1 = c_fc.weight.data        # [in_dim, hidden_dim]
        b1 = c_fc.bias.data          # [hidden_dim]

        in_dim = W1.size(0)
        hidden_dim = W1.size(1)

        # importance per hidden neuron (column) using L1 norm + bias
        scores = W1.abs().sum(dim=0) + b1.abs()
        k = max(1, int(hidden_dim * keep_ratio))
        keep_idx = torch.topk(scores, k=k, largest=True).indices.sort().values

        # Slice c_fc
        W1_new = W1[:, keep_idx]     # [in_dim, k]
        b1_new = b1[keep_idx]        # [k]

        # c_proj.weight: shape (hidden_dim, out_dim)
        W2 = c_proj.weight.data      # [hidden_dim, out_dim]
        b2 = c_proj.bias.data        # [out_dim]

        W2_new = W2[keep_idx, :]     # [k, out_dim]

        # Create new Conv1D layers with reduced hidden size
        k_hidden = W1_new.size(1)
        out_dim = W2_new.size(1)

        new_c_fc = Conv1D(k_hidden, in_dim).to(W1.device)
        new_c_proj = Conv1D(out_dim, k_hidden).to(W2.device)

        # Assign pruned weights
        new_c_fc.weight.data.copy_(W1_new)
        new_c_fc.bias.data.copy_(b1_new)
        new_c_proj.weight.data.copy_(W2_new)
        new_c_proj.bias.data.copy_(b2)

        # Replace modules in MLP
        mlp.c_fc = new_c_fc
        mlp.c_proj = new_c_proj

        # Optional: print per-layer info
        # print(f"Block {block_idx}: hidden_dim {hidden_dim} -> {k_hidden}")


# ---- structured MLP pruning ----
keep_ratio = 0.7  # keep 70% of neurons
print(f"Pruning MLP hidden neurons with keep_ratio={keep_ratio} ...")
prune_mlp_neurons(model, keep_ratio=keep_ratio)

print(f"New MLP hidden dim: {get_mlp_hidden_dim(model)}")

# OPTIONAL: after pruning, move to float16 on GPU for speed
if device.type == "cuda":
    model.to(dtype=torch.float16)

# ---- sanity-check generation ----
prompt = "Explain in one sentence how structured MLP pruning works."
inputs = tok(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )

print("Sample output:", tok.decode(out[0], skip_special_tokens=True))

# ---- AFTER pruning ----
after = bench_generation(
    model,
    tok,
    prompt="Why should we prune MLP neurons?",
    iters=20,
    max_new_tokens=20,
)
print("After structured MLP pruning:", after)

