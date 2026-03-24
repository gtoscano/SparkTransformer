from datasets import load_dataset
import evaluate
from transformers import pipeline


N_SAMPLES = 5
MODEL_ID = "facebook/bart-large-cnn"


def lead3_summary(text):
    sentences = text.split(". ")
    return ". ".join(sentences[:3]).strip()


def main():
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{N_SAMPLES}]")
    rouge = evaluate.load("rouge")
    summarizer = pipeline("summarization", model=MODEL_ID)

    references = dataset["highlights"]
    baseline_preds = []
    bart_preds = []

    for article in dataset["article"]:
        baseline_preds.append(lead3_summary(article))
        bart_preds.append(
            summarizer(article[:3000], max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
        )

    baseline_scores = rouge.compute(predictions=baseline_preds, references=references, use_stemmer=True)
    bart_scores = rouge.compute(predictions=bart_preds, references=references, use_stemmer=True)

    print("=== ROUGE: LEAD-3 BASELINE ===")
    print({k: round(v, 4) for k, v in baseline_scores.items()})

    print("\n=== ROUGE: BART ===")
    print({k: round(v, 4) for k, v in bart_scores.items()})

    print("\nQuestion for students: Does the higher ROUGE summary also look better to a human reader?")


if __name__ == "__main__":
    main()
