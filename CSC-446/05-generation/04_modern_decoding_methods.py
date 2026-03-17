"""
04_modern_decoding_methods.py

Compare newer or more practical decoding options in Hugging Face Transformers:
- top-p sampling baseline
- repetition penalty
- contrastive search
- typical sampling

Examples:
  python 04_modern_decoding_methods.py --prompt "Write a short paragraph about renewable energy:" --model gpt2
  python 04_modern_decoding_methods.py --prompt "The future of AI in education is" --model gpt2
"""

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict

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


@dataclass
class DecodeConfig:
    name: str
    generate_kwargs: Dict[str, Any]


def supports_contrastive_search(model) -> bool:
    """
    Some transformers builds expose contrastive search differently, and some
    environments route it through optional community generation hooks.
    Skip it when the current model/runtime cannot support it cleanly.
    """
    return (
        hasattr(type(model), "_contrastive_search")
        or hasattr(type(model), "contrastive_search")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="HF model id")
    parser.add_argument("--prompt", type=str, default="The Catholic University of America is a")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print(f"[Device] {device.type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    configs = [
        DecodeConfig(
            name="Baseline top-p sampling",
            generate_kwargs=dict(
                do_sample=True,
                top_p=0.9,
                temperature=0.9,
            ),
        ),
        DecodeConfig(
            name="Sampling + repetition penalty",
            generate_kwargs=dict(
                do_sample=True,
                top_p=0.9,
                temperature=0.9,
                repetition_penalty=1.2,
            ),
        ),
        DecodeConfig(
            name="Contrastive search",
            generate_kwargs=dict(
                penalty_alpha=0.6,
                top_k=4,
            ),
        ),
        DecodeConfig(
            name="Typical sampling",
            generate_kwargs=dict(
                do_sample=True,
                typical_p=0.9,
                temperature=0.9,
            ),
        ),
    ]

    print("\n" + "=" * 80)
    print(f"PROMPT: {args.prompt!r}")
    print("=" * 80)

    for idx, cfg in enumerate(configs):
        if cfg.name == "Contrastive search" and not supports_contrastive_search(model):
            print(f"\n--- {cfg.name} ---")
            print("[Skipped] This installed transformers/runtime build does not expose contrastive search for this model.")
            print("[Settings] {cfg.generate_kwargs}")
            continue

        set_seed(args.seed + idx)
        print(f"\n--- {cfg.name} ---")
        try:
            start = time.time()
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                **cfg.generate_kwargs,
            )
            elapsed = time.time() - start
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            continuation = text[len(args.prompt):] if text.startswith(args.prompt) else text

            print(f"[Time] {elapsed:.2f}s")
            print(f"[Settings] {cfg.generate_kwargs}")
            print(f"[Output]\n{args.prompt}{continuation}")
        except Exception as exc:
            print(f"[Skipped] {type(exc).__name__}: {exc}")
            print(f"[Settings] {cfg.generate_kwargs}")

    print("\nSuggested discussion:")
    print("- Which method best avoids repetition while staying coherent?")
    print("- Does contrastive search feel more stable than standard sampling?")
    print("- When does a repetition penalty help, and when does it distort the text?")


if __name__ == "__main__":
    main()
