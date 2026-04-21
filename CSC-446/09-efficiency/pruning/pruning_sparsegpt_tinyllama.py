# pruning_sparsegpt_tinyllama.py
# It requires the sparegpt to work. Do not forget to set up the directory below
# git clone https://github.com/IST-DASLab/sparsegpt.git
import os
import sys
import io
import contextlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from latency_bench import bench_generation

# ---- Add sparsegpt repo ----
SPARSEGPT_DIR = "/workspace/pruning/sparsegpt"
if SPARSEGPT_DIR not in sys.path:
    sys.path.append(SPARSEGPT_DIR)

from sparsegpt import SparseGPT  # from sparsegpt.py


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ---- tokenizer ----
tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

# ---- model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,   # keep fp32 for SparseGPT math
).to(device)

model.config.eos_token_id = tok.eos_token_id
model.config.pad_token_id = tok.pad_token_id

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model: {model_name}, params: {n_params:.1f}M, dtype: {next(model.parameters()).dtype}, device: {next(model.parameters()).device}")


print("Baseline:", bench_generation(model, tok, "SparseGPT pruning?"))


# ---- calibration data ----
def get_calibration_batches(tokenizer, seq_len=128, batch_size=4, num_batches=8):
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industry.",
        "Pruning reduces unnecessary parameters.",
        "Large language models benefit from efficiency.",
    ]
    for i in range(num_batches):
        batch = [texts[i % len(texts)] for _ in range(batch_size)]
        out = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_len,
        ).to(device)
        yield out["input_ids"], out["attention_mask"]


# ---- SparseGPT pruning ----
def sparsegpt_prune_model(model, tokenizer, sparsity=0.5, max_samples=2048):
    """
    Apply SparseGPT to each Linear layer:
    - Use forward *pre*-hooks to capture inputs (X)
    - Use forward hooks to capture outputs (Y)
    - For each layer: solve sparse regression with SparseGPT
    """
    # 1) Collect all Linear layers
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers.append((name, module))

    print(f"Found {len(layers)} linear layers for pruning.")

    act_in = {}
    act_out = {}

    # input: forward *pre*-hook -> (module, input)
    def make_hook_in(name):
        def hook(module, input):
            # input is a tuple; take first element (batch,...,in_features)
            act_in[name] = input[0].detach()
        return hook

    # output: forward hook -> (module, input, output)
    def make_hook_out(name):
        def hook(module, input, output):
            act_out[name] = output.detach()
        return hook

    # 2) Register hooks
    hook_in_handles = []
    hook_out_handles = []
    for name, module in layers:
        hook_in_handles.append(module.register_forward_pre_hook(make_hook_in(name)))
        hook_out_handles.append(module.register_forward_hook(make_hook_out(name)))

    # 3) Run model on calibration batches to fill act_in / act_out
    print("Collecting activations for SparseGPT...")
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask in get_calibration_batches(tokenizer):
            model(input_ids=input_ids, attention_mask=attention_mask)

    # 4) Remove hooks
    for h in hook_in_handles:
        h.remove()
    for h in hook_out_handles:
        h.remove()

    print("Running SparseGPT pruning per layer...")

    # 5) Prune each Linear layer
    for name, module in layers:
        if "lm_head" in name:
            print(f"[INFO] Skipping {name} (lm_head).")
            continue
        if name not in act_in or name not in act_out:
            print(f"[WARN] No activations for layer {name}, skipping.")
            continue

        X = act_in[name]      # shape: [B, ..., in_features]
        Y = act_out[name]     # shape: [B, ..., out_features]

        # Flatten batch + sequence dims
        X = X.reshape(-1, module.in_features)
        Y = Y.reshape(-1, module.out_features)

        # Optionally subsample for memory
        if X.size(0) > max_samples:
            X = X[:max_samples]
            Y = Y[:max_samples]

        #print(f"Pruning layer {name}: X={tuple(X.shape)}, Y={tuple(Y.shape)}")

        sg = SparseGPT(module)
        sg.add_batch(X, Y)
        buf = io.StringIO() 
        with contextlib.redirect_stdout(buf):
            sg.fasterprune(
                sparsity,  # e.g., 0.5 → 50% of weights zeroed
                prunen=0,  # unstructured sparsity
                prunem=0,
            )
        sg.free()

    print("SparseGPT pruning complete.")


print("Running SparseGPT pruning with 50% sparsity (unstructured)...")
sparsegpt_prune_model(model, tok, sparsity=0.5, max_samples=2048)

print("After SparseGPT:", bench_generation(model, tok, "SparseGPT pruning?"))

# ---- Save model ----
SAVE_DIR = "tinyllama_sparsegpt_50"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tok.save_pretrained(SAVE_DIR)
print(f"Saved pruned model to: {SAVE_DIR}")

