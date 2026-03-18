from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
import evaluate
from tqdm import tqdm
import os

from hf_auth import get_hf_token

N_SAMPLES = 50
GEN_KW = dict(max_new_tokens=200, do_sample=False)

MODELS = {
    "DISLab/SummLlama3.2-3B": {
        "task": "text-generation",
        "prompt": "Summarize the following news article in 3–5 concise sentences:\n\n{article}\n\nSummary:",
        "split_on": "Summary:",
        "tokenizer": "DISLab/SummLlama3.2-3B",
    },
    "raaec/Meta-Llama-3.1-8B-Instruct-Summarizer": {
        "task": "text-generation",
        "prompt": "Summarize the following news article in 3–5 concise sentences:\n\n{article}\n\nSummary:",
        "split_on": "Summary:",
        "tokenizer": None,
    },
}

def build_pipe(model_id, task, tokenizer_id=None):
    token = get_hf_token(required=True)
    tok = None
    if tokenizer_id:
        tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True, token=token)
    return pipeline(task, model=model_id, tokenizer=tok, device_map="auto", token=token)

def generate_summary(pipe, article, prompt_tmpl, split_on):
    prompt = prompt_tmpl.format(article=article)
    out = pipe(prompt, **GEN_KW)
    text = out[0]["generated_text"]
    if split_on in text:
        text = text.split(split_on, 1)[-1]
    return text.strip()

def main():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    rouge = evaluate.load("rouge")
    bleu  = evaluate.load("bleu")

    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{N_SAMPLES}]")
    articles   = ds["article"]
    references = ds["highlights"]

    for model_id, cfg in MODELS.items():
        print(f"\n=== Evaluating {model_id} on test[:{N_SAMPLES}] ===")
        pipe = build_pipe(model_id, cfg["task"], cfg["tokenizer"])

        preds = []
        for art in tqdm(articles, ncols=80):
            preds.append(generate_summary(pipe, art, cfg["prompt"], cfg["split_on"]))

        # ROUGE (strings)
        r = rouge.compute(predictions=preds, references=references, use_stemmer=True)
        print("ROUGE:", {k: round(v, 4) for k, v in r.items()})

        # BLEU (tokenized)
        pred_tok = [p.split() for p in preds]
        ref_tok  = [[r.split()] for r in references]
        b = bleu.compute(predictions=preds, references=[[ref] for ref in references])
        print("BLEU:", round(float(b["bleu"]), 4))

if __name__ == "__main__":
    main()
