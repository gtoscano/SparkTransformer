import argparse
import re

# python prompt_improver_inference.py --goal "Create a prompt to teach attention mechanism to beginners"

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def normalize_line(text: str) -> str:
    text = " ".join(text.split())
    # Collapse immediate repeated words like "clear, clear"
    return re.sub(r"\b(\w+)([\s,.;:]+)\1\b", r"\1", text, flags=re.IGNORECASE)


def cleanup_generation(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    parts = [p.strip(" -\t") for p in re.split(r"\s+-\s+", text) if p.strip(" -\t")]
    if len(parts) <= 1:
        return normalize_line(text)

    unique = []
    seen = set()
    for part in parts:
        line = normalize_line(part)
        key = line.lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(line)

    return "\n".join(f"- {line}" for line in unique)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate improved prompts.")
    parser.add_argument("--model_dir", default="./prompt-improver-t5-small")
    parser.add_argument(
        "--goal",
        default="I need a prompt that asks an assistant to explain backpropagation for beginners.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=120)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.eval()

    prompt = (
        "Rewrite and improve this prompt so it is specific, actionable, and clear.\n"
        f"User goal: {args.goal.strip()}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=12,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.35,
            no_repeat_ngram_size=4,
        )

    improved = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    improved = cleanup_generation(improved)
    if not improved:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=12,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                top_k=80,
                repetition_penalty=1.25,
                no_repeat_ngram_size=3,
            )
        improved = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        improved = cleanup_generation(improved)

    print("\nOriginal goal:")
    print(args.goal)
    print("\nImproved prompt:")
    print(improved)


if __name__ == "__main__":
    main()
