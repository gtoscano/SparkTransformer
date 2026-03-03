# 🧪 Lab Exercise 2 — Comparing Monolingual vs Multilingual Training
# Author: Dr. Gregorio Toscano
# --------------------------------------------------------------

import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

# --------------------------------------------------------------
# 1️⃣ Load Monolingual Datasets
# --------------------------------------------------------------
langs = ["de", "fr", "it", "en"]
datasets_lang = {}
results = {}

print("Loading PAN-X datasets...")
for lang in langs:
    ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
    datasets_lang[lang] = ds
    results[lang] = len(ds["train"])
print("Training samples per language:", results)

# --------------------------------------------------------------
# 2️⃣ Visualize Dataset Sizes
# --------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.bar(results.keys(), results.values(), color="#c41e3a")
plt.title("Training Samples per Language (PAN-X)")
plt.xlabel("Language")
plt.ylabel("Number of Training Sentences")
plt.grid(axis="y", alpha=0.3)
plt.show()

# --------------------------------------------------------------
# 3️⃣ Tokenization Helper
# --------------------------------------------------------------
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# --------------------------------------------------------------
# 4️⃣ Prepare Multilingual Dataset (German + French + Italian)
# --------------------------------------------------------------
print("Combining multilingual datasets...")
multi_train = concatenate_datasets([
    datasets_lang["de"]["train"],
    datasets_lang["fr"]["train"],
    datasets_lang["it"]["train"]
])
multi_valid = concatenate_datasets([
    datasets_lang["de"]["validation"],
    datasets_lang["fr"]["validation"],
    datasets_lang["it"]["validation"]
])

# Tokenize multilingual dataset
tokenized_multi_train = multi_train.map(tokenize_and_align_labels, batched=True)
tokenized_multi_valid = multi_valid.map(tokenize_and_align_labels, batched=True)

# --------------------------------------------------------------
# 5️⃣ Define Model and Training Arguments
# --------------------------------------------------------------
label_list = datasets_lang["de"]["train"].features["ner_tags"].feature.names
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
)

data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="xlmr-multilingual",
    evaluation_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=1,
    push_to_hub=False,
)

# --------------------------------------------------------------
# 6️⃣ Compute Metrics Function
# --------------------------------------------------------------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [
        [label_list[l] for l in label if l != -100] for label in labels
    ]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }

# --------------------------------------------------------------
# 7️⃣ Train Multilingual Model
# --------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_multi_train,
    eval_dataset=tokenized_multi_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Training multilingual model...")
trainer.train()

# --------------------------------------------------------------
# 8️⃣ Evaluate on Each Language (Zero-Shot)
# --------------------------------------------------------------
eval_scores = {}

for lang in langs:
    tokenized_eval = datasets_lang[lang]["validation"].map(
        tokenize_and_align_labels, batched=True
    )
    results = trainer.evaluate(tokenized_eval)
    eval_scores[lang] = results["eval_f1"]
    print(f"F1 for {lang}: {results['eval_f1']:.3f}")

# --------------------------------------------------------------
# 9️⃣ Visualize Monolingual vs Multilingual Performance
# --------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.bar(eval_scores.keys(), eval_scores.values(), color="#c41e3a")
plt.title("Zero-Shot F1 Scores — Trained Multilingually")
plt.xlabel("Evaluation Language")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.show()

# --------------------------------------------------------------
# ✅ End of Lab
# --------------------------------------------------------------
print("\n✅ Lab complete — compared monolingual vs multilingual performance!")
