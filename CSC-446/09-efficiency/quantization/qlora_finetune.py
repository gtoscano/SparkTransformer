from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

model_name = "meta-llama/Llama-3.2-1B-Instruct"
#model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "Qwen/Qwen2.5-1.5B"
#model_name = "mistralai/Mistral-7B-Instruct-v0.3"

print("Loading tokenizer/model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Important for padding
tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantization config (QLoRA)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_cfg,
    device_map="auto",
)
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA on top of quantized base
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_cfg)

print("Loading dataset...")
dataset = load_dataset("imdb")

def preprocess(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    # 🔑 Trainer will only compute loss if labels are present
    enc["labels"] = enc["input_ids"].copy()
    return enc

print("Tokenizing...")
dataset = dataset.map(preprocess, batched=True)

# Keep only the fields we need and make sure labels are there
columns_to_keep = ["input_ids", "attention_mask", "labels"]
dataset = dataset.remove_columns(
    [c for c in dataset["train"].column_names if c not in columns_to_keep]
)

# Very explicit: only these columns go into Torch tensors
dataset.set_format(type="torch", columns=columns_to_keep)

# (Optional) sanity check: uncomment to see keys in one example
# print(dataset["train"][0])

training_args = TrainingArguments(
    output_dir="./qlora-out",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    gradient_accumulation_steps=4,
    fp16=True,
    logging_steps=10,
    report_to="none",
    save_strategy="no",
)

# 🔑 Tell Trainer explicitly which field is the label field
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].select(range(500)),
)

print("Training QLoRA...")
trainer.train()
print("Saving QLoRA adapters...")
model.save_pretrained("./qlora-out")
tokenizer.save_pretrained("./qlora-out")

print("Done!")

