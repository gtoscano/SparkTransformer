from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from transformers.utils import logging
logging.set_verbosity_error()

dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
sample = dataset["train"][1]
article =  sample["article"]


model_id = "thepowerfuldeez/Qwen2-1.5B-Summarize"

# Use the official tokenizer (has all needed files)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

# Load the fine-tuned weights
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

# Decoder-only models: use text-generation with a summarization prompt
pipe = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")

prompt = f"""Summarize the following news article in 3–5 concise sentences:

{article}

Summary:"""
out = pipe(prompt, max_new_tokens=200, do_sample=False, temperature=0.2, truncation=True)
print(out[0]["generated_text"].split("Summary:")[-1].strip())
