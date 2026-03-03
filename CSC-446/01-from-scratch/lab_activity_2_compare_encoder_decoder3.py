# lab_activity_2_compare_encoder_decoder.py
# Compare Encoder (BERT fine-tune classification) vs Decoder (GPT-2 generation)
# - Fine-tunes BERT on a small IMDB subset
# - Generates text with GPT-2 under different decoding controls (warnings silenced + correct masking)
# - Reports speed, context size, accuracy, and prints a comparison table

import os, time, math, random, statistics
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertForSequenceClassification,
    GPT2LMHeadModel,
    set_seed,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
SEED = 42
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Tiny-ish settings for quick classroom runs
TRAIN_SAMPLES = 2000     # IMDB train subset
TEST_SAMPLES  = 400      # IMDB test subset
MAX_LEN = 192
BATCH_SIZE = 16
EPOCHS = 2
LR = 3e-5  # you can try 2e-5 + weight_decay=0.01 for stability

# Models
BERT_CKPT = "bert-base-uncased"
GPT2_CKPT = "gpt2"  # small gpt2 (context ~1024)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def format_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    return f"{m}m {s - 60*m:.0f}s"

def count_generated_tokens(seqs: torch.Tensor, prompt_len: int) -> int:
    # Rough count of newly generated tokens (per batch's first sample)
    return max(seqs.shape[-1] - prompt_len, 0)

@dataclass
class Benchmark:
    bert_train_time: float
    bert_eval_time: float
    bert_acc: float
    bert_infer_latency_ms: float
    gpt2_gen_latency_ms: float
    gpt2_tokens_generated: int
    gpt2_tokens_per_sec: float
    bert_context: int
    gpt2_context: int

def print_table(rows: List[Tuple[str, str]]):
    max_key = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{k:<{max_key}} : {v}")

# -----------------------------------------------------------------------------
# 1) Fine-tune BERT for classification
# -----------------------------------------------------------------------------
print("\n[1/3] Loading IMDB dataset (subset) …")
imdb = load_dataset("imdb")

train_small = imdb["train"].shuffle(seed=SEED).select(range(TRAIN_SAMPLES))
test_small  = imdb["test"].shuffle(seed=SEED).select(range(TEST_SAMPLES))

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_CKPT, use_fast=True)

def bert_tokenize_batch(batch):
    return bert_tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

train_small = train_small.map(bert_tokenize_batch, batched=True, remove_columns=["text"])
test_small  = test_small.map(bert_tokenize_batch, batched=True, remove_columns=["text"])

cols = ["input_ids", "attention_mask", "label"]
train_small.set_format(type="torch", columns=cols)
test_small.set_format(type="torch", columns=cols)

train_loader = DataLoader(train_small, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_small, batch_size=BATCH_SIZE)

print("[1/3] Initializing BERT classifier …")
bert_config = AutoConfig.from_pretrained(BERT_CKPT, num_labels=2)
bert_model = BertForSequenceClassification.from_pretrained(BERT_CKPT, config=bert_config).to(DEVICE)
bert_optimizer = torch.optim.AdamW(bert_model.parameters(), lr=LR)

def train_bert_one_epoch() -> float:
    bert_model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["label"].to(DEVICE)

        bert_optimizer.zero_grad()
        out = bert_model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        bert_optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def eval_bert() -> Tuple[float, float, List[int], List[int]]:
    bert_model.eval()
    total_loss = 0.0
    preds, gts = [], []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["label"].to(DEVICE)
        out = bert_model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        total_loss += out.loss.item() * input_ids.size(0)
        p = out.logits.argmax(dim=-1)
        preds.extend(p.cpu().tolist())
        gts.extend(labels.cpu().tolist())
    acc = accuracy_score(gts, preds)
    return total_loss / len(test_loader.dataset), acc, gts, preds

print("[1/3] Fine-tuning BERT …")
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    tr_loss = train_bert_one_epoch()
    val_loss, val_acc, gts, preds = eval_bert()
    print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
bert_train_time = time.time() - t0
print("BERT training time:", format_seconds(bert_train_time))
print("BERT test report:\n", classification_report(gts, preds, target_names=["neg","pos"], digits=4))

@torch.no_grad()
def measure_infer_latency(samples=10):
    bert_model.eval()
    latencies = []
    dummy = test_small[0]
    input_ids = dummy["input_ids"].unsqueeze(0).to(DEVICE)
    attn_mask = dummy["attention_mask"].unsqueeze(0).to(DEVICE)
    # warmup
    _ = bert_model(input_ids=input_ids, attention_mask=attn_mask)
    for _ in range(samples):
        t1 = time.time()
        _ = bert_model(input_ids=input_ids, attention_mask=attn_mask)
        latencies.append((time.time() - t1) * 1000.0)
    return statistics.median(latencies)

bert_infer_latency_ms = measure_infer_latency()
print(f"BERT single-example inference latency (median of 10): {bert_infer_latency_ms:.1f} ms")

# -----------------------------------------------------------------------------
# 2) Generate text with GPT-2 (decoder), benchmark speed & controls
#    (Warnings silenced + correct masking)
# -----------------------------------------------------------------------------
print("\n[2/3] Initializing GPT-2 …")
gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_CKPT, use_fast=True)
# ✅ GPT-2 has no pad token by default; set pad=eos to avoid warnings
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_CKPT).to(DEVICE)
gpt2_model.eval()

# Context size (respect model’s context window)
gpt2_context = int(getattr(
    gpt2_model.config, "n_positions",
    getattr(gpt2_model.config, "max_position_embeddings", 1024)
))
bert_context  = int(getattr(bert_model.config, "max_position_embeddings", 512))

prompt = (
    "In a surprising turn of events, the research team discovered that "
    "transformers could learn complex patterns with remarkable efficiency. "
    "Their experiments showed"
)

@torch.no_grad()
def timed_generate(prompt: str, max_new_tokens=80, **gen_kwargs) -> Dict[str, Any]:
    """
    Generate with GPT-2 while passing attention_mask and a valid pad_token_id.
    This silences warnings and ensures reliable behavior/timing.
    """
    tokens = gpt2_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=gpt2_context,   # ✅ respect context window
        padding=False,             # single prompt → no padding
    )
    input_ids = tokens["input_ids"].to(DEVICE)
    # Explicit attention mask (all ones for single, unpadded prompt)
    attention_mask = torch.ones_like(input_ids, device=DEVICE)

    prompt_len = input_ids.shape[-1]

    # Warmup to avoid first-iteration overhead in timing
    _ = gpt2_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=4,
        do_sample=False,
        pad_token_id=gpt2_tokenizer.eos_token_id,
    )

    t1 = time.time()
    out = gpt2_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,                # ✅ pass mask
        max_new_tokens=max_new_tokens,
        pad_token_id=gpt2_tokenizer.eos_token_id,     # ✅ valid pad id
        **gen_kwargs,
    )
    latency = (time.time() - t1)
    gen_tokens = count_generated_tokens(out, prompt_len)
    tps = gen_tokens / max(latency, 1e-6)
    text = gpt2_tokenizer.decode(out[0], skip_special_tokens=True)
    return {"latency_s": latency, "tokens": gen_tokens, "tps": tps, "text": text}

print("[2/3] GPT-2 baseline (greedy) …")
gpt2_greedy = timed_generate(prompt, max_new_tokens=100, do_sample=False)

print("[2/3] GPT-2 sampling (top-k=50, top-p=0.9, temperature=0.8) …")
gpt2_sampled = timed_generate(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    temperature=0.8,
    repetition_penalty=1.05,
)

# Style/Control demo via prompt engineering
control_prompt = (
    "Write a concise executive summary in a professional tone:\n\n"
    "Topic: Ethical use of AI in business\n\nSummary:"
)
print("[2/3] GPT-2 control via prompting (top-p=0.92, temperature=0.7) …")
gpt2_control = timed_generate(
    control_prompt,
    max_new_tokens=120,
    do_sample=True,
    top_p=0.92,
    temperature=0.7,
)

print("\n--- GPT-2 Outputs (first ~300 chars) ---")
print("[Greedy]", gpt2_greedy["text"][:300].replace("\n", " "))
print("\n[Sampled]", gpt2_sampled["text"][:300].replace("\n", " "))
print("\n[Prompt-Controlled]", gpt2_control["text"][:300].replace("\n", " "))

# -----------------------------------------------------------------------------
# 3) Compare speed, context, control
# -----------------------------------------------------------------------------
print("\n[3/3] Final evaluation of BERT (to time evaluation explicitly) …")
t_eval0 = time.time()
val_loss, val_acc, _, _ = eval_bert()
bert_eval_time = time.time() - t_eval0

bench = Benchmark(
    bert_train_time=bert_train_time,
    bert_eval_time=bert_eval_time,
    bert_acc=val_acc,
    bert_infer_latency_ms=bert_infer_latency_ms,
    gpt2_gen_latency_ms=gpt2_greedy["latency_s"] * 1000.0,
    gpt2_tokens_generated=gpt2_greedy["tokens"],
    gpt2_tokens_per_sec=gpt2_greedy["tps"],
    bert_context=bert_context,
    gpt2_context=gpt2_context,
)

print("\n==================== Comparison: Encoder vs Decoder ====================")
rows = [
    ("Task archetype",               "Encoder = Classification (supervised) | Decoder = Generation (LM)"),
    ("Model checkpoints",            f"{BERT_CKPT} | {GPT2_CKPT}"),
    ("Context length (tokens)",      f"BERT: {bench.bert_context} | GPT-2: {bench.gpt2_context}"),
    ("Training time (BERT)",         format_seconds(bench.bert_train_time)),
    ("Eval time (BERT val set)",     format_seconds(bench.bert_eval_time)),
    ("Accuracy (IMDB subset)",       f"{bench.bert_acc:.3f}"),
    ("BERT single-infer latency",    f"{bench.bert_infer_latency_ms:.1f} ms"),
    ("GPT-2 gen latency (100 tok)",  f"{bench.gpt2_gen_latency_ms:.1f} ms"),
    ("GPT-2 tokens generated",       f"{bench.gpt2_tokens_generated}"),
    ("GPT-2 tokens/sec",             f"{bench.gpt2_tokens_per_sec:.1f}"),
    ("Output control (decoder)",     "top-k/top-p/temperature/repetition_penalty + prompt engineering"),
    ("Output control (encoder)",     "n/a (classification head decides class distribution)"),
]
print_table(rows)
print("=======================================================================\n")

print("💬 Discussion Prompts")
print("1) When would you prefer an encoder model vs a decoder model?")
print("2) How does context length limit downstream tasks (long docs, code)?")
print("3) What trade-offs do sampling controls (temperature, top-k, top-p) introduce?")
print("4) How does supervised fine-tuning (BERT) differ from prompt engineering (GPT-2)?")
print("5) For the same hardware, where is the time spent: supervised training vs autoregressive decoding?")

print("\n✅ Done.")


