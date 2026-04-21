# latency_bench.py

import time
import torch

@torch.inference_mode()
def bench_generation(model, tok, prompt="Hello world!", iters=20, max_new_tokens=20):
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,   # 👈 IMPORTANT
    )

    # warmup
    model.generate(**inputs, **gen_kwargs)

    start = time.time()
    for _ in range(iters):
        model.generate(**inputs, **gen_kwargs)
    end = time.time()

    avg_ms = (end - start) / iters * 1000
    return {
        "avg_ms": round(avg_ms, 2),
        "device": str(device),
        "iters": iters,
        "tokens": max_new_tokens,
    }

