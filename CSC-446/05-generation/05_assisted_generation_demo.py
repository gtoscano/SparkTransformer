"""
05_assisted_generation_demo.py

Demonstrate assisted generation in Hugging Face Transformers.
This is a practical classroom proxy for the speculative decoding slide:
- run a target model alone
- run the same target model with an assistant model
- compare runtime and output

Examples:
  python 05_assisted_generation_demo.py --prompt "The Catholic University of America is preparing students for a future where AI" --target_model distilgpt2 --assistant_model sshleifer/tiny-gpt2
  python 05_assisted_generation_demo.py --prompt "The Catholic University of America is preparing students for a future where AI" --target_model gpt2-medium --assistant_model gpt2
"""

import argparse
import re
import time

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


def decode_output(tokenizer, prompt: str, output_ids) -> str:
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt):]
    return text


def clean_for_display(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_once(model, tokenizer, prompt_inputs, max_new_tokens: int, assistant_model=None):
    start = time.time()
    kwargs = dict(
        **prompt_inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if assistant_model is not None:
        kwargs["assistant_model"] = assistant_model
    output_ids = model.generate(**kwargs)
    elapsed = time.time() - start
    return output_ids, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_model",
        type=str,
        default="distilgpt2",
        help="HF model id for the larger/target model",
    )
    parser.add_argument(
        "--assistant_model",
        type=str,
        default="sshleifer/tiny-gpt2",
        help="HF model id for the smaller assistant model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The Catholic University of America is preparing students for a future where AI",
        help="Prompt for generation",
    )
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help="Discourage repeated 3-grams to keep the demo readable",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Mild repetition penalty for cleaner outputs",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print(f"[Device] {device.type}")
    print(f"[Target model] {args.target_model}")
    print(f"[Assistant model] {args.assistant_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    ensure_pad_token(tokenizer)

    target_model = AutoModelForCausalLM.from_pretrained(args.target_model).to(device)
    target_model.eval()

    assistant_model = AutoModelForCausalLM.from_pretrained(args.assistant_model).to(device)
    assistant_model.eval()

    prompt_inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    print("\n" + "=" * 80)
    print(f"PROMPT: {args.prompt!r}")
    print("=" * 80)

    set_seed(args.seed)
    baseline_ids, baseline_time = generate_once(
        target_model,
        tokenizer,
        {
            **prompt_inputs,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "repetition_penalty": args.repetition_penalty,
        },
        args.max_new_tokens,
    )
    baseline_text = clean_for_display(decode_output(tokenizer, args.prompt, baseline_ids))
    baseline_new_tokens = baseline_ids.shape[1] - prompt_inputs["input_ids"].shape[1]
    baseline_tok_per_s = baseline_new_tokens / max(baseline_time, 1e-8)

    print("\n--- Target model only ---")
    print(f"[Time] {baseline_time:.2f}s")
    print(f"[Generated tokens] {baseline_new_tokens}")
    print(f"[Tokens/sec] {baseline_tok_per_s:.2f}")
    print(f"[Output]\n{args.prompt}{baseline_text}")

    print("\n--- Assisted generation ---")
    try:
        set_seed(args.seed)
        assisted_ids, assisted_time = generate_once(
            target_model,
            tokenizer,
            {
                **prompt_inputs,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "repetition_penalty": args.repetition_penalty,
            },
            args.max_new_tokens,
            assistant_model=assistant_model,
        )
        assisted_text = clean_for_display(decode_output(tokenizer, args.prompt, assisted_ids))
        assisted_new_tokens = assisted_ids.shape[1] - prompt_inputs["input_ids"].shape[1]
        assisted_tok_per_s = assisted_new_tokens / max(assisted_time, 1e-8)
        speedup = baseline_time / max(assisted_time, 1e-8)

        if speedup > 1.05:
            verdict = "Assisted generation was faster."
        elif speedup < 0.95:
            verdict = "Assisted generation was slower."
        else:
            verdict = "Assisted generation and baseline were about the same speed."

        print(f"[Time] {assisted_time:.2f}s")
        print(f"[Generated tokens] {assisted_new_tokens}")
        print(f"[Tokens/sec] {assisted_tok_per_s:.2f}")
        print(f"[Approx speedup] {speedup:.2f}x")
        print(f"[Verdict] {verdict}")
        print(f"[Output]\n{args.prompt}{assisted_text}")
    except Exception as exc:
        print(f"[Skipped] {type(exc).__name__}: {exc}")
        print(
            "This transformers/runtime setup may not support assistant_model generation "
            "for the chosen models."
        )

    print("\nTeaching note:")
    print("- Assisted generation uses a smaller model to propose tokens and a target model to verify them.")
    print("- If supported, it can reduce latency while keeping output close to the target model.")
    print("- Token throughput and wall-clock time are the main quantities to compare in this demo.")
    print(
        f"- This demo uses no_repeat_ngram_size={args.no_repeat_ngram_size} "
        f"and repetition_penalty={args.repetition_penalty} to reduce visible loops."
    )


if __name__ == "__main__":
    main()
