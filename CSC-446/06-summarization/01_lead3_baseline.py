import nltk
from datasets import load_dataset
from nltk.tokenize import sent_tokenize


nltk.download("punkt", quiet=True)


def lead3_summary(text, n_sentences=3):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:n_sentences])


def main():
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test[:1]")
    sample = dataset[0]

    article = sample["article"]
    reference = sample["highlights"]
    prediction = lead3_summary(article)

    print("=== ARTICLE (first 1200 chars) ===\n")
    print(article[:1200] + "...")

    print("\n=== LEAD-3 BASELINE SUMMARY ===\n")
    print(prediction)

    print("\n=== REFERENCE HIGHLIGHTS ===\n")
    print(reference)

    print("\nQuestion for students: Why is this simple baseline often strong on news articles?")


if __name__ == "__main__":
    main()
