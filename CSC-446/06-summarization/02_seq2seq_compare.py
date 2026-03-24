from transformers import pipeline


MODELS = {
    "t5-small": "summarization",
    "facebook/bart-large-cnn": "summarization",
}

GEN_KW = {
    "max_length": 120,
    "min_length": 40,
    "do_sample": False,
}


def load_text():
    with open("cua.txt", "r", encoding="utf-8") as handle:
        return handle.read().strip()


def main():
    text = load_text()[:3000]

    print("=== SOURCE TEXT (first 800 chars) ===\n")
    print(text[:800] + "...")

    for model_id, task in MODELS.items():
        print(f"\n{'=' * 80}\nMODEL: {model_id}\n{'=' * 80}")
        summarizer = pipeline(task, model=model_id)
        output = summarizer(text, **GEN_KW)[0]["summary_text"]
        print("\nSUMMARY:\n")
        print(output)

    print("\nPrompt for discussion: Which model is more concise? Which one sounds more natural?")


if __name__ == "__main__":
    main()
