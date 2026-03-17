"""
02_analyze_generation_scores.py

Generate multiple continuations for a prompt and compare them using:
- continuation log-probability
- average log-probability per generated token
- perplexity derived from the continuation log-probability

This script is designed to support the slide sections on:
- Log Probabilities
- Log-Prob vs Quality
- Evaluation
- Perplexity

Examples:
  python 02_analyze_generation_scores.py --prompt "The Catholic University of America is a" --model gpt2
  python 02_analyze_generation_scores.py --prompt "Explain entropy in plain English in one short paragraph for undergraduate students:" --model gpt2 --num_samples 4
"""

import argparse
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List

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
class GenerationConfig:
    name: str
    generate_kwargs: Dict[str, Any]


def continuation_metrics(
    model,
    prompt_input_ids: torch.Tensor,
    full_output_ids: torch.Tensor,
) -> Dict[str, float]:
    """Compute continuation log-probability, avg log-probability, and perplexity."""
    input_len = prompt_input_ids.shape[1]
    with torch.no_grad():
        outputs = model(full_output_ids)
        logits = outputs.logits[:, :-1, :]
        labels = full_output_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        continuation = chosen[:, input_len - 1:]
        total_logprob = continuation.sum().item()
        num_generated_tokens = continuation.shape[1]
        avg_logprob = total_logprob / max(num_generated_tokens, 1)
        perplexity = math.exp(-avg_logprob)
        return {
            "total_logprob": total_logprob,
            "avg_logprob": avg_logprob,
            "perplexity": perplexity,
            "num_generated_tokens": num_generated_tokens,
        }


def readability_heuristic(text: str) -> Dict[str, float]:
    """
    A lightweight classroom-friendly heuristic.
    Higher is better. It rewards complete-ish prose and penalizes obvious
    degeneration such as repeated lines, numbering runs, and code-like artifacts.
    """
    stripped = text.strip()
    words = re.findall(r"\b[\w'-]+\b", stripped)
    word_count = len(words)
    unique_ratio = len(set(w.lower() for w in words)) / max(word_count, 1)
    sentences = re.split(r"[.!?]+", stripped)
    sentence_count = len([s for s in sentences if s.strip()])

    repeated_bigram_penalty = 0
    lowered_words = [w.lower() for w in words]
    bigrams = list(zip(lowered_words, lowered_words[1:]))
    if bigrams:
        repeated_bigram_penalty = len(bigrams) - len(set(bigrams))

    number_penalty = len(re.findall(r"\b\d+\b", stripped))
    code_penalty = 0
    for marker in ["./", "{", "}", "=", "->", "::", "import ", "def ", "class "]:
        if marker in stripped:
            code_penalty += 2

    colon_penalty = stripped.count(":")
    line_break_penalty = max(stripped.count("\n") - 1, 0)
    very_short_penalty = 8 if word_count < 20 else 0
    no_sentence_penalty = 6 if sentence_count == 0 else 0

    score = (
        30
        + 18 * unique_ratio
        + min(sentence_count, 4) * 2
        - repeated_bigram_penalty * 2.5
        - number_penalty * 0.75
        - code_penalty
        - colon_penalty * 0.5
        - line_break_penalty * 2
        - very_short_penalty
        - no_sentence_penalty
    )

    return {
        "readability_score": score,
        "unique_ratio": unique_ratio,
        "sentence_count": sentence_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="HF model id")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "In one short paragraph, explain entropy in plain English for undergraduate students. "
            "Use an everyday example and avoid equations."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
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
    results: List[Dict[str, Any]] = []

    configs = [
        GenerationConfig(
            name="Greedy baseline",
            generate_kwargs=dict(do_sample=False),
        )
    ]

    for idx in range(args.num_samples):
        configs.append(
            GenerationConfig(
                name=f"Sample {idx + 1}",
                generate_kwargs=dict(
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                ),
            )
        )

    print("\n" + "=" * 80)
    print(f"PROMPT: {args.prompt!r}")
    print("=" * 80)

    for idx, cfg in enumerate(configs):
        if cfg.generate_kwargs.get("do_sample"):
            set_seed(args.seed + idx)

        start = time.time()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **cfg.generate_kwargs,
        )
        elapsed = time.time() - start
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if text.startswith(args.prompt):
            continuation = text[len(args.prompt):]
        else:
            continuation = text

        metrics = continuation_metrics(model, inputs["input_ids"], output_ids)
        readability = readability_heuristic(continuation)

        print(f"\n--- {cfg.name} ---")
        print(
            f"[Time] {elapsed:.2f}s   "
            f"[Tokens] {metrics['num_generated_tokens']}   "
            f"[Cont. log-prob] {metrics['total_logprob']:.2f}   "
            f"[Avg log-prob/token] {metrics['avg_logprob']:.3f}   "
            f"[Perplexity] {metrics['perplexity']:.2f}   "
            f"[Readability heuristic] {readability['readability_score']:.2f}"
        )
        print(f"[Output]\n{args.prompt}{continuation}")
        results.append(
            {
                "name": cfg.name,
                "continuation": continuation,
                "metrics": metrics,
                "readability": readability,
            }
        )

    best_by_logprob = max(results, key=lambda r: r["metrics"]["total_logprob"])
    best_by_readability = max(results, key=lambda r: r["readability"]["readability_score"])

    print("\n" + "=" * 80)
    print("AUTOMATIC SUMMARY")
    print("=" * 80)
    print(
        "Best by log-probability: "
        f"{best_by_logprob['name']} "
        f"(cont. log-prob={best_by_logprob['metrics']['total_logprob']:.2f}, "
        f"perplexity={best_by_logprob['metrics']['perplexity']:.2f})"
    )
    print(
        "Best by readability heuristic: "
        f"{best_by_readability['name']} "
        f"(readability={best_by_readability['readability']['readability_score']:.2f})"
    )
    print(
        "Teaching note: these two winners may differ, which is the point of the demo."
    )

    print("\nInterpretation:")
    print("- Higher continuation log-probability means the model finds the text more likely.")
    print("- Lower perplexity means the continuation is less surprising to the model.")
    print("- The readability heuristic is only a rough proxy for human judgment, not a real evaluation metric.")
    print("- Compare all three views against the text itself to discuss quality versus likelihood.")


if __name__ == "__main__":
    main()
