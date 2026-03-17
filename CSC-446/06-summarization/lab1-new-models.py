from datasets import load_dataset
from transformers import pipeline
import evaluate
import re

# -----------------------
# Config (change if needed)
# -----------------------
TEST_SPLIT = "test[:50]"  # keep small for quick run; e.g. "test[:200]" for more
MAX_NEW_TOKENS = 200
MIN_NEW_TOKENS = 40       # a little length control
DO_SAMPLE = False

MODELS = [
    ("DISLab/SummLlama3.2-3B", "seq2seq"),
    ("raaec/Meta-Llama-3.1-8B-Instruct-Summarizer", "decoder"),
]


def clean_text(s: str) -> str:
    s = s.replace("<n>", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # Sometimes instruct models echo "Summary:" – strip anything before it
    if "Summary:" in s:
        s = s.split("Summary:", 1)[-1].strip()
    return s


def main():
    # Load data
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=TEST_SPLIT)
    articles = ds["article"]
    refs = [clean_text(x) for x in ds["highlights"]]

    # Metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

    for model_id, _family in MODELS:
        print(f"\n=== Evaluating {model_id} on {TEST_SPLIT} ===")
        pipe = pipeline(
            "summarization",
            model=model_id,
            device_map="auto",
        )

        preds = []
        for i, art in enumerate(articles, 1):
            out = pipe(
                art,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                truncation=True,
            )[0]["summary_text"]
            preds.append(clean_text(out))
            if i % 10 == 0:
                print(f"  {i}/{len(articles)}")

        # Compute metrics
        r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
        b = bleu.compute(predictions=preds, references=[refs])

        print(
            f"ROUGE-1: {r['rouge1']:.4f} | "
            f"ROUGE-2: {r['rouge2']:.4f} | "
            f"ROUGE-L: {r['rougeL']:.4f} | "
            f"BLEU: {b['score']:.4f}"
        )


if __name__ == "__main__":
    main()

