import numpy as np
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
    processing_class=tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# 4) Train & evaluate
trainer.train()
print(trainer.evaluate())

# 5) Inference example
pred = trainer.predict(imdb_test.select(range(3)))
print("Pred labels:", np.argmax(pred.predictions, axis=-1).tolist())

