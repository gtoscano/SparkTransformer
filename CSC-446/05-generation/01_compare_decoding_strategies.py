"""
01_compare_decoding_strategies.py

Demo of text generation decoding strategies with Hugging Face Transformers:
- Greedy
- Beam search
- Sampling: (A) top-k only, (B) top-p only, (C) top-k + top-p

Run:
  python 01_compare_decoding_strategies.py --prompt "The Catholic University of America is a" --model gpt2 --max_new_tokens 80
  python 01_compare_decoding_strategies.py --prompt "The Catholic University of America is a" --model mistralai/Mistral-7B-v0.1  --max_new_tokens 200

Notes:
  - Uses a small model by default (gpt2) for quick classroom demos.
  - If you have a stronger GPU, try: --model gpt2-xl
  - Reproducible via torch.manual_seed and generation 'generator'.


# 1) Quick CPU demo
python 01_compare_decoding_strategies.py --prompt "The Catholic University of America is a" --model gpt2 --max_new_tokens 80

# 2) More creative prompt + different sampling feel
python 01_compare_decoding_strategies.py --prompt "Write the opening paragraph of a sci-fi short story set on Europa:" --model gpt2 --max_new_tokens 120

# 3) If you have a GPU and time
python 01_compare_decoding_strategies.py --prompt "Explain the concept of entropy to a high-schooler:" --model gpt2-xl --max_new_tokens 120


# 🧠 **Top 5 Models for Text Generation (Hugging Face Ecosystem)**

| Model                                    | Org / Size                  | Description & Strengths                                                                                                                                                                   | Command Example                              |
| ---------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| **`gpt2` / `gpt2-medium` / `gpt2-xl`**   | OpenAI (117 M–1.5 B params) | Classic baseline. Small, fast, perfect for teaching decoding strategies. Still produces fluent English text.                                                                              | `--model gpt2-xl`                            |
| **`EleutherAI/gpt-neo-2.7B`**            | EleutherAI (2.7 B)          | GPT-2-style architecture trained on *The Pile* (825 GB). Larger vocab, better coherence. Great next step up if you have a GPU with > 12 GB VRAM.                                          | `--model EleutherAI/gpt-neo-2.7B`            |
| **`EleutherAI/gpt-j-6B`**                | EleutherAI (6 B)            | A strong open-source GPT-3-class model. Excellent for creative writing, Q&A, and general reasoning. Often runs fine on > 16 GB GPUs or quantized CPU builds.                              | `--model EleutherAI/gpt-j-6B`                |
| **`EleutherAI/gpt-neox-20b`**            | EleutherAI (20 B)           | Much more fluent and consistent than GPT-J; a research-grade open GPT-3-scale model. Heavy — ~40 GB VRAM required — but good for cluster demos.                                           | `--model EleutherAI/gpt-neox-20b`            |
| **`mistralai/Mistral-7B-Instruct-v0.2`** | Mistral AI (7 B)            | Modern, instruction-tuned model with excellent efficiency. Performs well even on consumer GPUs using `torch.float16` or quantized versions. Great “modern GPT-3.5-class” model for demos. | `--model mistralai/Mistral-7B-Instruct-v0.2` |

### ⚙️ **Recommendations**

* **For classroom laptops / Colab** → `gpt2` or `gpt2-medium` (fast and safe).
* **For GPU labs (A100, RTX 3090, etc.)** → `EleutherAI/gpt-neo-2.7B` or `Mistral-7B`.
* **For advanced research nodes** → `gpt-j-6B` or `gpt-neox-20b`.
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------- Utilities ----------

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_pad_token(tokenizer):
    # GPT-2 family has no pad_token by default; align pad with eos for generation APIs
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DecodeConfig:
    name: str
    generate_kwargs: Dict[str, Any]


def sequence_logprob(model, input_ids: torch.Tensor, full_output_ids: torch.Tensor, input_len: int) -> float:
    """
    Compute total log-probability of the generated continuation (excluding prompt tokens).
    Works for causal LMs.
    """
    with torch.no_grad():
        # Forward on labels = full sequence to align logits/labels for next-token prediction
        out = model(full_output_ids)
        logits = out.logits[:, :-1, :]                  # [B, L-1, V]
        labels = full_output_ids[:, 1:]                 # [B, L-1]
        # Select the log-prob of the actually chosen tokens
        log_probs = torch.log_softmax(logits, dim=-1)   # [B, L-1, V]
        chosen = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        # Only sum over the generated continuation (not the prompt)
        cont_logprob = chosen[:, input_len-1:].sum().item()
        return cont_logprob

def generate_and_score(model, tokenizer, device, prompt: str, cfg: DecodeConfig,
                       max_new_tokens: int, seed: int):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    start = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        **cfg.generate_kwargs,
    )
    elapsed = time.time() - start
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    cont_logp = sequence_logprob(model, inputs["input_ids"], output_ids, input_len=input_len)
    return text, elapsed, cont_logp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="HF model id (e.g., gpt2, gpt2-medium, gpt2-xl)")
    parser.add_argument("--prompt", type=str, default="The Catholic University of America is the", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="New tokens to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beams", type=int, default=5, help="num_beams for beam search")
    parser.add_argument("--no_repeat_ngram", type=int, default=2, help="no_repeat_ngram_size for beam search")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print(f"[Device] {device.type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    # ----- Decoding configurations -----
    configs = [
        DecodeConfig(
            name="Greedy",
            generate_kwargs=dict(do_sample=False)  # deterministic
        ),
        DecodeConfig(
            name=f"Beam (num_beams={args.beams}, no_repeat_ngram_size={args.no_repeat_ngram})",
            generate_kwargs=dict(
                do_sample=False,
                num_beams=args.beams,
                no_repeat_ngram_size=args.no_repeat_ngram,
                early_stopping=True
            )
        ),
        DecodeConfig(
            name="Sampling A (top-k=50, top-p=1.0, temp=0.9)",
            generate_kwargs=dict(
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9
            )
        ),
        DecodeConfig(
            name="Sampling B (top-k=0, top-p=0.9, temp=0.9)  # nucleus only",
            generate_kwargs=dict(
                do_sample=True,
                top_k=0,          # disable top-k to emphasize top-p
                top_p=0.9,
                temperature=0.9
            )
        ),
        DecodeConfig(
            name="Sampling C (top-k=50, top-p=0.9, temp=0.9)  # combined",
            generate_kwargs=dict(
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.9
            )
        ),
    ]

    print("\n" + "="*80)
    print(f"PROMPT: {args.prompt!r}")
    print("="*80 + "\n")

    for cfg in configs:
        text, elapsed, cont_logp = generate_and_score(
            model, tokenizer, device,
            prompt=args.prompt, cfg=cfg,
            max_new_tokens=args.max_new_tokens, seed=args.seed
        )
        # For display, show only the continuation after the prompt for clarity
        if text.startswith(args.prompt):
            continuation = text[len(args.prompt):]
        else:
            continuation = text

        print(f"--- {cfg.name} ---")
        print(f"[Time] {elapsed:.2f}s   [Continuation log-prob] {cont_logp:.2f}")
        print(f"[Output]\n{args.prompt}{continuation}\n")
        print("-"*80)

    print("Tip: Try changing temperature (e.g., 0.7 vs 1.3) and top_k/top_p to feel creativity vs. coherence.")
    print("     For longer outputs, increase --max_new_tokens. For stronger models, try --model gpt2-xl.")


if __name__ == "__main__":
    main()
