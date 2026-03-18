from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from hf_auth import get_hf_token
from transformers.utils import logging

logging.set_verbosity_error()

dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
sample = dataset["train"][1]
article =  sample["article"]


model_id = "thepowerfuldeez/Qwen2-1.5B-Summarize"
token = get_hf_token(required=True)

# Use the official tokenizer (has all needed files)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", token=token)

# Load the fine-tuned weights
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto", token=token)

# Decoder-only models: use text-generation with a summarization prompt
pipe = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto", token=token)

prompt = f"""Summarize the following news article in 3–5 concise sentences:

{article}

Summary:"""
out = pipe(prompt, max_new_tokens=200, do_sample=False, temperature=0.2, truncation=True)
print(out[0]["generated_text"].split("Summary:")[-1].strip())
