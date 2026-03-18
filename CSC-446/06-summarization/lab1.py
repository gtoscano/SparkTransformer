import os
from transformers import AutoTokenizer, pipeline

from hf_auth import get_hf_token

# Disable HF telemetry
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Models
MODELS = {
    "DISLab/SummLlama3.2-3B": {
        "task": "text-generation",
        "prompt": "Summarize the following text in 3–5 concise sentences:\n\n{text}\n\nSummary:",
        "split_on": "Summary:",
        "tokenizer": "DISLab/SummLlama3.2-3B",
    },
    "raaec/Meta-Llama-3.1-8B-Instruct-Summarizer": {
        "task": "text-generation",
        "prompt": "Summarize the following text in 3–5 concise sentences:\n\n{text}\n\nSummary:",
        "split_on": "Summary:",
        "tokenizer": None,
    },
}

GEN_KW = dict(max_new_tokens=250, do_sample=False, temperature=0.2)


def build_pipe(model_id, task, tokenizer_id=None):
    token = get_hf_token(required=True)
    tok = None
    if tokenizer_id:
        tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True, token=token)
    return pipeline(task, model=model_id, tokenizer=tok, device_map="auto", token=token)


def generate_summary(pipe, text, prompt_tmpl, split_on):
    prompt = prompt_tmpl.format(text=text)
    out = pipe(prompt, **GEN_KW)
    text = out[0]["generated_text"]
    if split_on in text:
        text = text.split(split_on, 1)[-1]
    return text.strip()


def main():
    # Load local cua.txt
    with open("cua.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()

    for model_id, cfg in MODELS.items():
        print(f"\n=== Generating summary with {model_id} ===")
        pipe = build_pipe(model_id, cfg["task"], cfg["tokenizer"])
        summary = generate_summary(pipe, text, cfg["prompt"], cfg["split_on"])
        print("\n--- Summary ---\n")
        print(summary)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
