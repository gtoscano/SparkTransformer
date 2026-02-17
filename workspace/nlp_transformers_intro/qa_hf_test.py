from transformers import pipeline


def main() -> None:
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
        "Name one NLP task where transformers are used.",
    ]

    print(f"Model: {model_id}\n")
    print(f"Context: {context}\n")
    for question in questions:
        result = qa(question=question, context=context)
        print(f"Q: {question}")
        print(f"A: {result['answer']} (score={result['score']:.4f})\n")


if __name__ == "__main__":
    main()
