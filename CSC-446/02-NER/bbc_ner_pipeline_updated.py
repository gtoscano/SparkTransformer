#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BBC Multilingual Headlines -> NER -> JSON/CSV + Chart

What this script does:
1) Scrapes BBC headlines from Spanish, and French portals.
2) Runs a multilingual NER model (XLM-R) to tag PER/ORG/LOC/MISC.
3) Aggregates entity counts and saves JSONL + CSV.
4) Plots a simple bar chart of total entities per type.

Usage:
  python bbc_ner_pipeline.py --limit 40 --outdir outputs
  # Optional (force device): --device cpu|cuda|mps|auto
  # Optional (offline mode with local HTML):
  python bbc_ner_pipeline.py --local_html bbc_en.html bbc_es.html bbc_fr.html

Dependencies (recommended):
  pipenv install requests beautifulsoup4 lxml pandas matplotlib transformers torch sentencepiece protobuf
  # If you ever see a tiktoken path in an error:
  pipenv install tiktoken

Notes:
- Be respectful: limit requests & follow robots.txt.
- For classroom demo, --limit keeps things snappy.
"""

from __future__ import annotations
import sys
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Network & parsing
import requests
from bs4 import BeautifulSoup

# Data / plotting
import pandas as pd
import matplotlib.pyplot as plt

# NER
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

BBC_SITES = [
    ("es", "https://www.bbc.com/mundo"),
    ("fr", "https://www.bbc.com/afrique"),
]

# Multilingual NER model (solid & relatively lightweight)
DEFAULT_MODEL = "Davlan/xlm-roberta-base-ner-hrl"


def fetch_headlines(url: str, limit: int = 40, delay: float = 0.5) -> List[str]:
    """Fetch headlines (h3) from a BBC landing page."""
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # BBC often uses <h3> for teasers; this keeps the demo simple.
    items = []
    for h in soup.find_all("h3"):
        t = h.get_text(" ", strip=True)
        if t and len(t.split()) >= 3:  # avoid overly short labels
            items.append(t)
        if len(items) >= limit:
            break

    # Gentle politeness delay
    time.sleep(delay)
    return items


def parse_local_html(path: Path, limit: int = 40) -> List[str]:
    """Parse headlines from a local HTML copy (offline mode)."""
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    items = []
    for h in soup.find_all("h3"):
        t = h.get_text(" ", strip=True)
        if t and len(t.split()) >= 3:
            items.append(t)
        if len(items) >= limit:
            break
    return items


def pick_device(name: str | None) -> str:
    """Resolve device selection."""
    if name and name.lower() != "auto":
        return name
    if hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_ner(model_name: str = DEFAULT_MODEL, device_str: str = "auto"):
    """
    Load NER with a robust tokenizer fallback.
    1) Try fast tokenizer (Rust); if it fails,
    2) Fallback to slow tokenizer (SentencePiece)
    """
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e_fast:
        print("[WARN] Fast tokenizer failed, retrying with slow (SentencePiece). Error:\n", e_fast)
        try:
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except Exception as e_slow:
            raise SystemExit(
                "Tokenizer load failed. Try: `pip install sentencepiece tokenizers protobuf tiktoken`\n"
                f"Fast error: {e_fast}\nSlow error: {e_slow}"
            )

    mdl = AutoModelForTokenClassification.from_pretrained(model_name)
    dev = pick_device(device_str)
    print(f"Using device: {dev}")

    # transformers pipeline: device=0 -> CUDA GPU 0; device=-1 -> CPU
    # (MPS currently maps via CPU pathway in pipeline's device arg; use -1)
    pipe_device = 0 if dev == "cuda" else -1

    ner = pipeline(
        "token-classification",
        model=mdl,
        tokenizer=tok,
        aggregation_strategy="simple",  # merges subwords to word-level spans
        device=pipe_device,
    )
    return ner


def _safe_int(x, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def run_ner(ner, texts: List[str], lang: str) -> List[Dict[str, Any]]:
    """Run NER over a list of texts; return flat list of entity dicts with metadata."""
    out: List[Dict[str, Any]] = []
    for idx, txt in enumerate(texts):
        try:
            ents = ner(txt)
        except Exception:
            ents = []
        for ent in ents:
            out.append({
                "lang": lang,
                "headline_index": idx,
                "headline": txt,
                "entity_text": ent.get("word"),
                "entity_type": ent.get("entity_group"),
                "start": _safe_int(ent.get("start", -1)),
                "end": _safe_int(ent.get("end", -1)),
                "score": float(ent.get("score", 0.0)),
            })
    return out


def aggregate_counts(entities: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(entities)
    if df.empty:
        return pd.DataFrame(columns=["lang", "entity_type", "count"])
    agg = (
        df.groupby(["lang", "entity_type"])
          .size()
          .reset_index(name="count")
          .sort_values(["lang", "count"], ascending=[True, False])
    )
    return agg


def plot_entity_totals(entities: List[Dict[str, Any]], outdir: Path):
    df = pd.DataFrame(entities)
    if df.empty:
        print("No entities to plot.")
        return
    totals = (
        df.groupby("entity_type")
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(totals["entity_type"], totals["count"])
    ax.set_title("Total Entities by Type (All Languages)")
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Count")
    fig.tight_layout()
    outpath = outdir / "entity_totals.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved chart: {outpath}")


def main():
    ap = argparse.ArgumentParser(description="BBC Headlines -> Multilingual NER Classroom Pipeline")
    ap.add_argument("--limit", type=int, default=40, help="Max headlines per site")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF model for NER")
    ap.add_argument("--local_html", nargs="*", help="Optional local HTML files in order: en, es, fr")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda", "mps"],
                    help="Computation device")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect headlines
    dataset: List[Dict[str, Any]] = []
    if args.local_html:
        # Offline mode: user supplies local HTML files
        lang_paths = list(zip(["en", "es", "fr"], [Path(p) for p in args.local_html]))
        for lang, p in lang_paths:
            if not p.exists():
                print(f"[WARN] Local HTML not found: {p}")
                headlines = []
            else:
                headlines = parse_local_html(p, limit=args.limit)
            for h in headlines:
                dataset.append({"lang": lang, "headline": h})
    else:
        # Online scraping (be respectful)
        for lang, url in BBC_SITES:
            try:
                headlines = fetch_headlines(url, limit=args.limit)
            except Exception as e:
                print(f"[WARN] Failed to fetch {url}: {e}")
                headlines = []
            for h in headlines:
                dataset.append({"lang": lang, "headline": h})

    # Save raw headlines
    df_raw = pd.DataFrame(dataset)
    raw_csv = outdir / "bbc_headlines.csv"
    df_raw.to_csv(raw_csv, index=False)
    print(f"Saved raw headlines: {raw_csv}  (rows={len(df_raw)})")

    if df_raw.empty:
        print("No headlines collected; exiting.")
        return

    # Load NER
    print(f"Loading NER model: {args.model}")
    ner = load_ner(args.model, args.device)

    # Run NER language by language for clearer progress
    entities: List[Dict[str, Any]] = []
    for lang in df_raw["lang"].unique():
        sub = df_raw[df_raw["lang"] == lang]["headline"].tolist()
        ents = run_ner(ner, sub, lang)
        entities.extend(ents)
        print(f"Processed NER for lang={lang}: {len(ents)} entities")

    # Save entities as JSONL and CSV
    jsonl_path = outdir / "bbc_entities.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for e in entities:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Saved entities JSONL: {jsonl_path}  (lines={len(entities)})")

    df_ent = pd.DataFrame(entities)
    ent_csv = outdir / "bbc_entities.csv"
    df_ent.to_csv(ent_csv, index=False)
    print(f"Saved entities CSV: {ent_csv}")

    # Aggregate & save counts
    agg = aggregate_counts(entities)
    agg_csv = outdir / "bbc_entity_counts.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"Saved entity counts CSV: {agg_csv}")

    # Plot totals across languages
    plot_entity_totals(entities, outdir)

    print("✅ Done. Explore the outputs directory for CSV/JSONL and the chart.")


if __name__ == "__main__":
    main()

