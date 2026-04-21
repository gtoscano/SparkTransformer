# pruning_unstructured_example.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils import prune
from latency_bench import bench_generation 

model_name = "Qwen/Qwen2.5-1.5B"

# 1) Load tokenizer
tok = AutoTokenizer.from_pretrained(model_name)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

# 2) Load model on a single GPU (simpler than device_map for this example)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,   # new name instead of torch_dtype
).cuda()

# Make sure model config also knows about pad/eos
model.config.eos_token_id = tok.eos_token_id
model.config.pad_token_id = tok.pad_token_id

# ---- baseline latency BEFORE pruning ----
baseline = bench_generation(model, tok, "Why is pruning useful?")
print("Baseline:", baseline)


# 3) Prune 30% of weights in all linear layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.30)

# 4) Permanently remove masks (make pruning final)
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.remove(module, "weight")

print("Pruning complete.")

# 5) Run a small generation to test
prompt = "Explain why pruning can speed up inference."
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tok.pad_token_id,   # <--- important
    )

print(tok.decode(out[0], skip_special_tokens=True))


# ---- latency AFTER pruning ----
after = bench_generation(model, tok, "Why is pruning useful?")
print("After pruning:", after)
