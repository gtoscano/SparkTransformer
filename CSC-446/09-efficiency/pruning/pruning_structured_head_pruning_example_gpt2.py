# pruning_structured_example_gpt2.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# ---- model (load in float32, THEN prune, THEN maybe half) ----
model = AutoModelForCausalLM.from_pretrained(model_name)  # float32
model.to(device)

# ensure config matches tokenizer
model.config.eos_token_id = tok.eos_token_id
model.config.pad_token_id = tok.pad_token_id

# quick helper to inspect heads after pruning
def get_heads_per_layer(m):
    # use first block's attention module as reference
    attn = m.transformer.h[0].attn
    return attn.num_heads if hasattr(attn, "num_heads") else m.config.n_head

print(f"Original heads per layer: {get_heads_per_layer(model)}")

# ---- baseline BEFORE pruning ----
baseline = bench_generation(
    model,
    tok,
    prompt="Why should we prune attention heads?",
    iters=20,
    max_new_tokens=20,
)
print("Baseline:", baseline)

# ---- structured head pruning ----
# heads_to_prune[layer_idx] = [head indices]
heads_to_prune = {i: [0, 1, 2, 3] for i in range(model.config.n_layer)}
print("Pruning heads", heads_to_prune[0], "in ALL layers...")

# This performs **structured head pruning**:
# it slices attention weights & updates internal head count.
model.transformer._prune_heads(heads_to_prune)

print(f"New heads per layer: {get_heads_per_layer(model)}")

# OPTIONAL: after pruning, move to float16 on GPU for speed
if device.type == "cuda":
    model.to(dtype=torch.float16)
    # tiny safety: make sure embeddings / lm_head follow
    # (model.to(...) already handles this globally)

# ---- sanity-check generation ----
prompt = "Explain in one sentence how structured head pruning works."
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
    prompt="Why should we prune attention heads?",
    iters=20,
    max_new_tokens=20,
)
print("After structured head pruning:", after)

