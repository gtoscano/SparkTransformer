from transformers import pipeline
# Read the file
with open("cua.txt", "r", encoding="utf-8") as f:
    text = f.read()
# Define models to test
models = ["t5-small", "facebook/bart-large-cnn", "google/pegasus-cnn_dailymail"]

# Run summarization for each model
for m in models:
    summarizer = pipeline("summarization", model=m, device=0)
    chunk = text[:2000]
    summary = summarizer(chunk, max_new_tokens=512, do_sample=False)[0]["summary_text"]
    print(f"\n--- {m} ---\n{summary}\n")

