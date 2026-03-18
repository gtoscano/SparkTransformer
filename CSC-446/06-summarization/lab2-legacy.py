#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Safe evaluator for 'traditional' summarization models on CNN/DailyMail.
Computes ROUGE and BLEU, while avoiding CUDA device-side asserts by:
  - Disabling flash/SDPA attention kernels (uses math attention)
  - Clamping tokenizer max length
  - Per-example CPU fallback if a CUDA error occurs

Usage:
  python lab2_legacy_safe.py

Customize:
  - MODELS list
  - N_SAMPLES
  - GEN_KW
"""

# ---- MUST be set before importing torch/transformers ----
import os
os.environ["PYTORCH_FORCE_SDP_ATTENTION"] = "0"   # avoid flash/mem-efficient kernels
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"          # cleaner stack traces if an error slips through

import sys
import json
from typing import List, Tuple

import torch
from datasets import load_dataset
from hf_auth import get_hf_token
from transformers import (
    AutoTokenizer,
    pipeline,
)
from tqdm import tqdm

# Metrics
import evaluate

# Try to disable SDPA kernels explicitly (older/newer PyTorch variants)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

# --------------------- Config ---------------------

# Traditional (encoder-decoder) summarizers you want to evaluate
MODELS = [
    "facebook/bart-large-cnn",
    "google/pegasus-cnn_dailymail",
    "t5-small",
]

# How many test samples to evaluate from CNN/DM
N_SAMPLES = 50

# Generation kwargs (kept conservative for comparability & stability)
GEN_KW = dict(
    max_new_tokens=200,
    do_sample=False,
    temperature=None,
    top_p=None,
    truncation=True,
)


# ------------------ Utilities --------------------

def build_summarizer(model_id: str, device: int):
    """
    Build a summarization pipeline with a tokenizer clamp to avoid
    extreme sequence lengths that can trigger SDPA/flash attention issues.
    """
    token = get_hf_token()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=token)
    # Keep encoder inputs <= 1024 tokens for safety (common for BART/PEGASUS)
    tok.model_max_length = min(getattr(tok, "model_max_length", 1024) or 1024, 1024)
    tok.truncation_side = "right"
    return pipeline("summarization", model=model_id, tokenizer=tok, device=device, token=token)


def trim_for_encoder(text: str, max_chars: int = 4000) -> str:
    """
    A light pre-trim to cut pathological very-long articles.
    Tokenizer will still apply token-level truncation.
    """
    if text and len(text) > max_chars:
        return text[:max_chars]
    return text


def safe_summarize(pipe, text: str, **gen_kw) -> str:
    """
    Run a single summarization; if a CUDA device-side assert occurs,
    retry on CPU so the run can continue.
    """
    try:
        return pipe(text, **gen_kw)[0]["summary_text"]
    except RuntimeError as e:
        msg = str(e).lower()
        # Known flaky path for some GPUs: fallback to CPU for this one article
        if "cuda error" in msg or "device-side assert" in msg:
            cpu_pipe = pipeline(
                "summarization",
                model=pipe.model,
                tokenizer=pipe.tokenizer,
                device=-1,
                token=get_hf_token(),
            )
            return cpu_pipe(text, **gen_kw)[0]["summary_text"]
        raise


def compute_rouge_bleu(preds: List[str], refs: List[str]) -> Tuple[dict, dict]:
    """
    Compute ROUGE and BLEU with the `evaluate` library.
    BLEU: pass raw strings; sacrebleu inside `evaluate` handles tokenization.
    ROUGE: defaults (rouge1/2/L/Lsum).
    """
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    # ROUGE expects: predictions: List[str], references: List[str]
    r = rouge.compute(predictions=preds, references=refs)

    # BLEU expects:
    # - predictions: List[str]
    # - references: List[List[str]] (list of reference strings per example)
    b = bleu.compute(predictions=preds, references=[[r] for r in refs])

    return r, b


# ------------------ Main Eval --------------------

def evaluate_on_cnndm(models: List[str], n_samples: int = 50):
    print(f"Loading CNN/DailyMail (cnn_dailymail 3.0.0) test[:{n_samples}] …")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{n_samples}]")

    # Extract articles and reference summaries
    articles = [ex["article"] for ex in dataset]
    refs = [ex["highlights"] for ex in dataset]

    device = 0 if torch.cuda.is_available() else -1

    for model_id in models:
        print(f"\n=== Evaluating {model_id} on test[:{n_samples}] ===")
        # Build pipeline with safe tokenizer settings
        try:
            pipe = build_summarizer(model_id, device)
        except Exception as e:
            print(f"⚠️  Could not load {model_id}: {e}")
            continue

        preds: List[str] = []
        for art in tqdm(articles, total=len(articles)):
            try:
                out = safe_summarize(pipe, trim_for_encoder(art), **GEN_KW)
            except Exception as e:
                # If something truly unexpected happens, record the skip and continue
                print(f"\n⚠️ Skipped due to: {e}\n")
                preds.append("")  # Keep indexing aligned
                continue
            preds.append(out.strip())

        # Compute metrics
        rouge_res, bleu_res = compute_rouge_bleu(preds, refs)

        # Pretty print
        print("ROUGE:", {k: float(v) for k, v in rouge_res.items()})
        print("BLEU :", {k: (float(v) if not isinstance(v, list) else v) for k, v in bleu_res.items()})

        # Optional: save raw outputs
        results_blob = {
            "model": model_id,
            "n_samples": n_samples,
            "rouge": {k: float(v) for k, v in rouge_res.items()},
            "bleu": {k: (float(v) if not isinstance(v, list) else v) for k, v in bleu_res.items()},
            "predictions": preds,
            "references": refs,
        }
        outname = model_id.replace("/", "__") + f"__cnndm_{n_samples}.json"
        with open(outname, "w", encoding="utf-8") as f:
            json.dump(results_blob, f, ensure_ascii=False, indent=2)
        print(f"Saved details to {outname}")


def main():
    evaluate_on_cnndm(MODELS, N_SAMPLES)


if __name__ == "__main__":
    main()
