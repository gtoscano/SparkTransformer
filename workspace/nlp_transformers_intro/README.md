# NLP with Transformers: First Hugging Face Examples

This folder is a minimal intro project for NLP with Hugging Face Transformers.

It includes:
- Sentiment analysis (classification)
- Translation (sequence-to-sequence)
- Extractive question answering (QA)
- Text generation (causal language modeling)
- Prompt improvement by fine-tuning on a Hugging Face prompt dataset

## Files

- `simple_hf_test.py`: first sentiment test script
- `translate_hf_test.py`: translation test script (English -> Spanish)
- `qa_hf_test.py`: question-answering test script
- `generation_hf_test.py`: text generation test script
- `train_prompt_improver.py`: fine-tune `t5-small` to rewrite prompts
- `prompt_improver_inference.py`: generate improved prompts with the fine-tuned model
- `requirements.txt`: dependencies

## Option 1: Run on your computer

### macOS / Linux

```bash
cd workspace/nlp_transformers_intro
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python simple_hf_test.py
python translate_hf_test.py
python qa_hf_test.py
python generation_hf_test.py
```

### Windows (PowerShell)

```powershell
cd workspace\nlp_transformers_intro
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python simple_hf_test.py
python translate_hf_test.py
python qa_hf_test.py
python generation_hf_test.py
```

## Prompt Improvement Fine-Tuning (Local)

This trains on the Hugging Face dataset `fka/awesome-chatgpt-prompts`.

```bash
cd workspace/nlp_transformers_intro
source .venv/bin/activate
python train_prompt_improver.py --max_train_samples 2000 --epochs 1
python prompt_improver_inference.py --goal "Create a prompt to teach attention mechanism to beginners"
```

If students have limited hardware, reduce training size:

```bash
python train_prompt_improver.py --max_train_samples 500 --batch_size 4 --epochs 1
```

## Option 2: Run on Google Colab

1. Open `https://colab.research.google.com/`
2. Install dependencies in the first cell:

```python
!pip -q install transformers torch datasets accelerate sentencepiece
```

### Colab sentiment example

```python
from transformers import pipeline

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_id)

examples = [
    "I love learning NLP with transformers.",
    "The movie was okay, but a bit too long.",
    "This homework is confusing and frustrating.",
]

for text in examples:
    result = classifier(text)[0]
    print(text)
    print(result)
```

### Colab translation example

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
model_id = "Helsinki-NLP/opus-mt-en-es"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

examples = [
    "Hello everyone, welcome to our NLP class.",
    "Transformers are powerful models for language tasks.",
]

for text in examples:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    output_ids = model.generate(**inputs, max_new_tokens=120)
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"EN: {text}")
    print(f"ES: {result}\n")
```

### Colab question-answering example

```python
from transformers import pipeline

model_id = "distilbert-base-cased-distilled-squad"
qa = pipeline("question-answering", model=model_id)

context = (
    "Transformers are neural network models introduced in 2017. "
    "They rely on self-attention to process relationships between tokens "
    "and are widely used in NLP tasks like translation, summarization, and QA."
)

questions = [
    "When were transformers introduced?",
    "What mechanism do transformers rely on?",
]

for q in questions:
    out = qa(question=q, context=context)
    print(f"Q: {q}")
    print(f"A: {out['answer']} (score={out['score']:.4f})\n")
```

### Colab text-generation example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.generation_config.pad_token_id = tokenizer.eos_token_id

prompts = [
    "In this NLP class, transformers are important because",
    "A simple explanation of attention is",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
    )
    out = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {out}\n")
```

### Colab fine-tuning + inference example

```python
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

model_id = "t5-small"
dataset_id = "fka/awesome-chatgpt-prompts"

def build_source_text(act_text: str) -> str:
    return (
        "Rewrite and improve this prompt so it is specific, actionable, and clear.\n"
        f"User goal: {act_text.strip()}"
    )

dataset = load_dataset(dataset_id, split="train")
dataset = dataset.filter(lambda x: x.get("act") is not None and x.get("prompt") is not None)
dataset = dataset.select(range(min(1000, len(dataset))))

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

def preprocess(batch):
    sources = [build_source_text(x) for x in batch["act"]]
    targets = [x.strip() for x in batch["prompt"]]
    model_inputs = tokenizer(sources, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=256, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
split = tokenized.train_test_split(test_size=0.1, seed=42)

training_args = Seq2SeqTrainingArguments(
    output_dir="./prompt-improver-t5-small",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    predict_with_generate=True,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
)

trainer.train()
trainer.save_model("./prompt-improver-t5-small")
tokenizer.save_pretrained("./prompt-improver-t5-small")
```

Inference cell after training:

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_dir = "./prompt-improver-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.eval()

goal = "Create a prompt that helps students understand gradient descent with a simple example."
query = (
    "Rewrite and improve this prompt so it is specific, actionable, and clear.\n"
    f"User goal: {goal}"
)

inputs = tokenizer(query, return_tensors="pt", truncation=True)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=120, num_beams=4)

print("Goal:")
print(goal)
print("\nImproved prompt:")
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Notes for Class

- This prompt-improver is a teaching example, not a production alignment pipeline.
- Results depend on data quality and model size.
- Students can switch datasets or models to compare performance.
