import argparse

# python prompt_improver_inference.py --goal "Create a prompt to teach attention mechanism to beginners"

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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
            num_beams=4,
            early_stopping=True,
        )

    improved = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nOriginal goal:")
    print(args.goal)
    print("\nImproved prompt:")
    print(improved)


if __name__ == "__main__":
    main()
