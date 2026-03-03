# 🧪 Multilingual NER Lab — Fine-Tune on German, Test on French
# Course: NLP with Transformers
# --------------------------------------------------------------

# !pip install datasets transformers seqeval accelerate -q

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import torch

# --------------------------------------------------------------
# 1️⃣ Load Dataset (PAN-X)
# --------------------------------------------------------------
print("Loading PAN-X datasets (German and French)...")
panx_de = load_dataset("xtreme", name="PAN-X.de")
panx_fr = load_dataset("xtreme", name="PAN-X.fr")

# --------------------------------------------------------------
# 2️⃣ Inspect Labels
# --------------------------------------------------------------
label_list = panx_de["train"].features["ner_tags"].feature.names
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}
print("Labels:", label_list)

# --------------------------------------------------------------
# 3️⃣ Tokenization and Alignment
# --------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

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

print("Tokenizing training and validation data...")
tokenized_panx_de = panx_de.map(tokenize_and_align_labels, batched=True)
tokenized_panx_fr = panx_fr.map(tokenize_and_align_labels, batched=True)

# --------------------------------------------------------------
# 4️⃣ Model Setup
# --------------------------------------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base", num_labels=len(label_list), id2label=id2label, label2id=label2id
)

# --------------------------------------------------------------
# 5️⃣ Training Arguments
# --------------------------------------------------------------
args = TrainingArguments(
    output_dir="xlmr-panx-de",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    push_to_hub=False,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

# --------------------------------------------------------------
# 6️⃣ Metrics
# --------------------------------------------------------------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[l] for l in label if l != -100]
        for label in labels
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
# 7️⃣ Trainer
# --------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_panx_de["train"],
    eval_dataset=tokenized_panx_de["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --------------------------------------------------------------
# 8️⃣ Training
# --------------------------------------------------------------
print("Training model on German data...")
trainer.train()

# --------------------------------------------------------------
# 9️⃣ Evaluation on German (in-language)
# --------------------------------------------------------------
results = trainer.evaluate()
print("Evaluation on German:", results)

# --------------------------------------------------------------
# 🔟 Zero-Shot Evaluation on French (cross-lingual)
# --------------------------------------------------------------
print("\nZero-shot testing on French sentence...")
ner_pipeline = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
)

text_fr = "Jeff Dean est informaticien chez Google en Californie."
output = ner_pipeline(text_fr)
for entity in output:
    print(f"{entity['word']:<15} {entity['entity_group']} ({entity['score']:.2f})")

# --------------------------------------------------------------
# ✅ End of Lab
# --------------------------------------------------------------
print("\n✅ Lab complete — you fine-tuned on German and tested on French!")
