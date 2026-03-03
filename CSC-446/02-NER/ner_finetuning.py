
import numpy as np
from datasets import load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

print("Loading PAN-X German dataset...")
panx_de = load_dataset("xtreme", name="PAN-X.de")

label_list = panx_de["train"].features["ner_tags"].feature.names
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    output_dir="xlmr-ner-de",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=1,
    push_to_hub=False,
)



def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,  # input is already word-tokenized
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)          # special tokens ([CLS], [SEP])
            elif word_id != prev_word_id:
                aligned.append(labels[word_id])  # first subtoken of a word
            else:
                aligned.append(-100)          # subsequent subtokens → ignore
            prev_word_id = word_id
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized

panx_de_encoded = panx_de.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=panx_de["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

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
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=panx_de_encoded["train"],
    eval_dataset=panx_de_encoded["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("xlmr-ner-de")

def get_predictions(trainer, dataset):
    output = trainer.predict(dataset)
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

print("\n--- German (in-domain) evaluation ---")
de_true, de_pred = get_predictions(trainer, panx_de_encoded["validation"])
print(classification_report(de_true, de_pred))

# Zero-shot evaluation on French
print("Loading PAN-X French dataset for zero-shot evaluation...")
panx_fr = load_dataset("xtreme", name="PAN-X.fr")
panx_fr_encoded = panx_fr["validation"].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=panx_fr["validation"].column_names,
)
print("\n--- French (zero-shot) evaluation ---")
fr_true, fr_pred = get_predictions(trainer, panx_fr_encoded)
print(classification_report(fr_true, fr_pred))

