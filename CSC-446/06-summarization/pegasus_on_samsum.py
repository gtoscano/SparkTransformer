from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, PegasusTokenizer,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer 
)

# 1) ids vs objects
model_id  = "google/pegasus-cnn_dailymail"
tokenizer = PegasusTokenizer.from_pretrained(model_id,  use_fast=False)       # requires `sentencepiece`
model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)  # <-- model OBJECT

# 2) data (SAMSum)
ds = load_dataset("knkarthick/samsum")


# 3) preprocess (warning-free)
def prep(batch):
    x = tokenizer(batch["dialogue"], max_length=1024, truncation=True)
    y = tokenizer(text_target=batch["summary"], max_length=128, truncation=True)
    x["labels"] = y["input_ids"]
    return x

tok = ds.map(prep, batched=True, remove_columns=ds["train"].column_names)

# 4) trainer bits
collator = DataCollatorForSeq2Seq(tokenizer, model=model)
args = Seq2SeqTrainingArguments(                     # <-- changed class
    output_dir="pegasus-samsum",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    predict_with_generate=True,                      # <-- now valid
    generation_max_length=128,
    report_to=[],
)

trainer = Seq2SeqTrainer(
    model=model,                     # <-- object, not string
    args=args,
    data_collator=collator,
    tokenizer=tokenizer,
    train_dataset=tok["train"],
    eval_dataset=tok["validation"]
)

trainer.train()
print(trainer.evaluate(tok["test"]))

