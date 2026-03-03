"""
Fairness Across Languages — NER Evaluation
===========================================
Loads the German-fine-tuned XLM-RoBERTa and evaluates it zero-shot on
multiple languages to answer:
  1. Does performance drop for minority/low-resource languages?
  2. Are certain entity types (PER, LOC, ORG) biased?
  3. Are scripts (Latin vs non-Latin) handled equally?

Requires: ner_finetuning.py to have been run first (produces xlmr-ner-de/).
"""

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------
# 1. Language metadata
#    (code, script, resource_level)
#    resource_level is approximate based on PAN-X training sizes
# ---------------------------------------------------------------
LANGUAGES = [
    ("de", "Latin",       "high"),   # training language
    ("en", "Latin",       "high"),
    ("fr", "Latin",       "high"),
    ("es", "Latin",       "high"),
    ("it", "Latin",       "high"),
    ("nl", "Latin",       "high"),
    ("pt", "Latin",       "mid"),
    ("pl", "Latin",       "mid"),
    ("tr", "Latin",       "mid"),
    ("id", "Latin",       "mid"),
    ("vi", "Latin",       "mid"),
    ("sw", "Latin",       "low"),
    ("tl", "Latin",       "low"),
    ("ar", "Arabic",      "mid"),
    ("fa", "Arabic",      "mid"),
    ("ur", "Arabic",      "low"),
    ("zh", "CJK",         "high"),
    ("ja", "CJK",         "mid"),
    ("ko", "CJK",         "mid"),
    ("hi", "Devanagari",  "mid"),
    ("mr", "Devanagari",  "low"),
    ("bn", "Bengali",     "low"),
    ("ru", "Cyrillic",    "high"),
    ("bg", "Cyrillic",    "mid"),
    ("el", "Greek",       "mid"),
    ("ka", "Georgian",    "low"),
    ("th", "Thai",        "mid"),
    ("my", "Myanmar",     "low"),
]

SCRIPT_COLORS = {
    "Latin":      "#4e79a7",
    "Arabic":     "#f28e2b",
    "CJK":        "#e15759",
    "Devanagari": "#76b7b2",
    "Bengali":    "#59a14f",
    "Cyrillic":   "#edc948",
    "Greek":      "#b07aa1",
    "Georgian":   "#ff9da7",
    "Thai":       "#9c755f",
    "Myanmar":    "#bab0ac",
}

ENTITY_TYPES = ["PER", "LOC", "ORG"]

# ---------------------------------------------------------------
# 2. Load fine-tuned model (from saved model or latest checkpoint)
# ---------------------------------------------------------------
MODEL_DIR = "xlmr-ner-de"

# If the root dir has no config.json, find the latest checkpoint subfolder
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    checkpoints = sorted(
        [d for d in os.listdir(MODEL_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No saved model or checkpoints found in '{MODEL_DIR}'. "
            "Run ner_finetuning.py first."
        )
    MODEL_DIR = os.path.join(MODEL_DIR, checkpoints[-1])

print(f"Loading fine-tuned model from '{MODEL_DIR}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

id2label = model.config.id2label
label_list = [id2label[i] for i in range(len(id2label))]
print(f"Labels: {label_list}\n")

# ---------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != prev_word_id:
                aligned.append(labels[word_id])
            else:
                aligned.append(-100)
            prev_word_id = word_id
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized


# Trainer used only for prediction (no training)
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="tmp_fairness_eval", no_cuda=False),
    tokenizer=tokenizer,
)


def get_predictions(encoded_dataset):
    output = trainer.predict(encoded_dataset)
    preds = np.argmax(output.predictions, axis=2)
    labels = output.label_ids
    true_labels = [
        [label_list[l] for l in label if l != -100] for label in labels
    ]
    true_preds = [
        [label_list[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    return true_labels, true_preds


# ---------------------------------------------------------------
# 4. Evaluate all languages
# ---------------------------------------------------------------
rows = []

for lang, script, resource in LANGUAGES:
    print(f"  [{lang}] script={script:12s} resource={resource:4s} ...", end=" ", flush=True)
    try:
        ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
        train_size = len(ds["train"])
        val = ds["validation"]
        encoded = val.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=val.column_names,
        )
        true, pred = get_predictions(encoded)
        report = classification_report(true, pred, output_dict=True)

        row = {
            "lang":      lang,
            "script":    script,
            "resource":  resource,
            "train_size": train_size,
            "precision": round(precision_score(true, pred), 4),
            "recall":    round(recall_score(true, pred), 4),
            "f1":        round(f1_score(true, pred), 4),
        }
        for etype in ENTITY_TYPES:
            row[f"{etype}_f1"] = round(report.get(etype, {}).get("f1-score", float("nan")), 4)

        rows.append(row)
        print(f"F1={row['f1']:.3f}  PER={row['PER_f1']:.3f}  LOC={row['LOC_f1']:.3f}  ORG={row['ORG_f1']:.3f}")

    except Exception as e:
        print(f"SKIPPED — {e}")

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError(
        "No languages were evaluated successfully. "
        "Check the SKIPPED messages above for details."
    )

df.to_csv("outputs/fairness_results.csv", index=False)

# ---------------------------------------------------------------
# 5. Print summary table
# ---------------------------------------------------------------
print("\n" + "=" * 75)
print("FAIRNESS SUMMARY")
print("=" * 75)
print(df.to_string(index=False, float_format="%.3f"))

# Per-script averages
print("\n--- Average F1 by Script ---")
print(df.groupby("script")["f1"].mean().sort_values(ascending=False).to_string(float_format="%.3f"))

# Per-resource-level averages
print("\n--- Average F1 by Resource Level ---")
print(df.groupby("resource")["f1"].mean().sort_values(ascending=False).to_string(float_format="%.3f"))

# ---------------------------------------------------------------
# 6. Plot 1 — F1 per language, colored by script
# ---------------------------------------------------------------
df_sorted = df.sort_values("f1", ascending=False)
colors = [SCRIPT_COLORS.get(s, "#aaaaaa") for s in df_sorted["script"]]

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(df_sorted["lang"], df_sorted["f1"], color=colors)

de_f1 = df.loc[df["lang"] == "de", "f1"].values
if len(de_f1):
    ax.axhline(de_f1[0], color="black", linestyle="--", linewidth=1.2,
               label=f"German baseline ({de_f1[0]:.3f})")
    ax.legend()

ax.set_title("Zero-Shot NER F1 per Language (model trained on German)")
ax.set_xlabel("Language")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.3)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SCRIPT_COLORS.values()]
ax.legend(handles + [plt.Line2D([0], [0], color="black", linestyle="--")],
          list(SCRIPT_COLORS.keys()) + ["German baseline"],
          title="Script", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("outputs/fairness_f1_per_lang.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 7. Plot 2 — Per-entity F1 heatmap across languages
# ---------------------------------------------------------------
entity_df = df[["lang"] + [f"{e}_f1" for e in ENTITY_TYPES]].set_index("lang")

fig, ax = plt.subplots(figsize=(6, len(df) * 0.38 + 1))
im = ax.imshow(entity_df.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(ENTITY_TYPES)))
ax.set_xticklabels(ENTITY_TYPES, fontsize=11)
ax.set_yticks(range(len(entity_df)))
ax.set_yticklabels(entity_df.index, fontsize=9)
plt.colorbar(im, ax=ax, label="F1 Score")
ax.set_title("Per-Entity F1 Across Languages\n(answers: are certain entity types biased?)")

for i in range(len(entity_df)):
    for j in range(len(ENTITY_TYPES)):
        val = entity_df.values[i, j]
        if not np.isnan(val):
            text_color = "black" if 0.25 < val < 0.85 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

plt.tight_layout()
plt.savefig("outputs/fairness_entity_heatmap.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 8. Plot 3 — Average F1 by script
# ---------------------------------------------------------------
script_f1 = df.groupby("script")["f1"].mean().sort_values()
script_colors_list = [SCRIPT_COLORS.get(s, "#aaaaaa") for s in script_f1.index]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(script_f1.index, script_f1.values, color=script_colors_list)
ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
ax.set_title("Average Zero-Shot F1 by Script\n(answers: are scripts handled equally?)")
ax.set_xlabel("Average F1 Score")
ax.set_xlim(0, 1.05)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/fairness_f1_by_script.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 9. Plot 4 — Training size vs F1 (minority language question)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
for script, group in df.groupby("script"):
    ax.scatter(group["train_size"], group["f1"],
               label=script, color=SCRIPT_COLORS.get(script, "#aaaaaa"),
               s=90, zorder=3)
    for _, row in group.iterrows():
        ax.annotate(row["lang"], (row["train_size"], row["f1"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)

ax.set_xscale("log")
ax.set_title("Training Size vs Zero-Shot F1\n(answers: do minority languages underperform?)")
ax.set_xlabel("PAN-X Training Set Size (log scale)")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1)
ax.legend(title="Script", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/fairness_size_vs_f1.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# Clean up temp dir
# ---------------------------------------------------------------
import shutil
if os.path.exists("tmp_fairness_eval"):
    shutil.rmtree("tmp_fairness_eval")

print("\n✅ Fairness evaluation complete.")
print("   outputs/fairness_results.csv")
print("   outputs/fairness_f1_per_lang.png")
print("   outputs/fairness_entity_heatmap.png")
print("   outputs/fairness_f1_by_script.png")
print("   outputs/fairness_size_vs_f1.png")
