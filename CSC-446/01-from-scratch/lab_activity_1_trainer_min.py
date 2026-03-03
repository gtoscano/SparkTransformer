import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer
)
import evaluate

MODEL = "distilbert-base-uncased"     # small & fast
MAX_LEN = 192
TRAIN_SAMPLES = 1000
TEST_SAMPLES  = 200
BATCH = 16
EPOCHS = 2
LR = 2e-5

# 1) Data
imdb = load_dataset("imdb")
imdb_train = imdb["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
imdb_test  = imdb["test"].shuffle(seed=42).select(range(TEST_SAMPLES))

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
def tokenize(ex):
    return tok(ex["text"], truncation=True, max_length=MAX_LEN)
imdb_train = imdb_train.map(tokenize, batched=True, remove_columns=["text"])
imdb_test  = imdb_test.map(tokenize,  batched=True, remove_columns=["text"])

collator = DataCollatorWithPadding(tokenizer=tok)
metric = evaluate.load("accuracy")

# 2) Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

# 3) Trainer
args = TrainingArguments(
    output_dir="out_trainer",
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=25,
    report_to="none",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=imdb_train,
    eval_dataset=imdb_test,
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# 4) Train & evaluate
trainer.train()
print(trainer.evaluate())
# --- Enable attentions on the fine-tuned model
trainer.model.config.output_attentions = True
trainer.model.config.return_dict = True

def viz_attention(
    text: str,
    layer: int = 0,
    head: int = 0,
    max_tokens: int = 40,
    save_path: str | None = None,
):
    """
    Plot a self-attention heatmap for DistilBERT after fine-tuning.
    - layer: 0..(num_hidden_layers-1)  (DistilBERT base: 6 layers)
    - head:  0..(num_attention_heads-1) (DistilBERT base: 12 heads)
    """
    # Tokenize and move to the same device as the model
    enc = tok(
        text,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(trainer.model.device) for k, v in enc.items()}

    # Forward pass with attentions
    with torch.no_grad():
        out = trainer.model(**enc, output_attentions=True, return_dict=True)

    # out.attentions: tuple of length = num_layers
    # each element shape: [batch, num_heads, seq_len, seq_len]
    att = out.attentions[layer][0, head]  # [T, T]
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])

    # Trim to keep the plot readable
    T = min(len(tokens), max_tokens)
    att = att[:T, :T].detach().cpu()

    # Plot
    plt.figure(figsize=(7, 6))
    plt.imshow(att, aspect="auto")
    plt.colorbar()
    plt.title(f"DistilBERT attention — layer {layer}, head {head}")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.xticks(range(T), tokens[:T], rotation=90)
    plt.yticks(range(T), tokens[:T])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved attention map to {save_path}")
    else:
        plt.show()

# --- Example usages:
viz_attention("time flies like an arrow", layer=0, head=0)
# or visualize on a real test example:
sample = imdb_test[0]["text"]
viz_attention(sample, layer=1, head=3, max_tokens=30, save_path="distilbert_attn_l1h3.png")

# 5) Inference example
pred = trainer.predict(imdb_test.select(range(3)))
print("Pred labels:", np.argmax(pred.predictions, axis=-1).tolist())

