"""
03_temperature_topk_topp_lab.py

Sweep temperature, top-k, and top-p settings for a fixed prompt.
This script is designed as a lab companion for the decoding slides.

Examples:
  python 03_temperature_topk_topp_lab.py --prompt "Write one paragraph about the moon:" --model gpt2
  python 03_temperature_topk_topp_lab.py --prompt "Explain entropy in plain English:" --temps 0.5 0.8 1.1
"""

import argparse
import time
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_list(values: Iterable[str], cast_type):
    return [cast_type(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="HF model id")
    parser.add_argument("--prompt", type=str, default="The Catholic University of America is a")
    parser.add_argument("--max_new_tokens", type=int, default=70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temps", nargs="*", default=["0.5", "0.8", "1.0", "1.3"])
    parser.add_argument("--top_ks", nargs="*", default=["20", "50"])
    parser.add_argument("--top_ps", nargs="*", default=["0.8", "0.9", "0.95"])
    args = parser.parse_args()

    temperatures: List[float] = parse_list(args.temps, float)
    top_ks: List[int] = parse_list(args.top_ks, int)
    top_ps: List[float] = parse_list(args.top_ps, float)

    set_seed(args.seed)
    device = pick_device()
    print(f"[Device] {device.type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    print("\n" + "=" * 80)
    print(f"PROMPT: {args.prompt!r}")
    print("=" * 80)

    print("\nTEMPERATURE SWEEP")
    for idx, temp in enumerate(temperatures):
        set_seed(args.seed + idx)
        start = time.time()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_k=0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        elapsed = time.time() - start
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        continuation = text[len(args.prompt):] if text.startswith(args.prompt) else text
        print(f"\n--- temperature={temp:.2f} ---")
        print(f"[Time] {elapsed:.2f}s")
        print(f"{args.prompt}{continuation}")

    print("\n" + "=" * 80)
    print("TOP-K SWEEP")
    for idx, top_k in enumerate(top_ks):
        set_seed(args.seed + 100 + idx)
        start = time.time()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_k=top_k,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        elapsed = time.time() - start
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        continuation = text[len(args.prompt):] if text.startswith(args.prompt) else text
        print(f"\n--- top_k={top_k} ---")
        print(f"[Time] {elapsed:.2f}s")
        print(f"{args.prompt}{continuation}")

    print("\n" + "=" * 80)
    print("TOP-P SWEEP")
    for idx, top_p in enumerate(top_ps):
        set_seed(args.seed + 200 + idx)
        start = time.time()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_k=0,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        elapsed = time.time() - start
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        continuation = text[len(args.prompt):] if text.startswith(args.prompt) else text
        print(f"\n--- top_p={top_p:.2f} ---")
        print(f"[Time] {elapsed:.2f}s")
        print(f"{args.prompt}{continuation}")

    print("\nSuggested discussion:")
    print("- Which temperature produces the best balance of coherence and diversity?")
    print("- Which top-k value is too restrictive or too noisy?")
    print("- Does top-p feel more adaptive than top-k for the same prompt?")


if __name__ == "__main__":
    main()
