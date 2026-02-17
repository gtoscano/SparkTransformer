from transformers import pipeline


def main() -> None:
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    classifier = pipeline("sentiment-analysis", model=model_id)

    examples = [
        "I love learning NLP with transformers.",
        "The movie was okay, but a bit too long.",
        "This homework is confusing and frustrating.",
    ]

    print(f"Model: {model_id}\n")
    for text in examples:
        result = classifier(text)[0]
        label = result["label"]
        score = result["score"]
        print(f"Text: {text}")
        print(f"Prediction: {label} ({score:.4f})\n")


if __name__ == "__main__":
    main()
