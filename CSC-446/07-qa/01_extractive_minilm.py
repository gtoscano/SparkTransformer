from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers.utils.logging import set_verbosity_error


set_verbosity_error()
MODEL_ID = "deepset/minilm-uncased-squad2"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)
    qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

    question = "Where does Professor Lin-Ching work?"
    context = "Professor Lin-Ching works at CUA in Washington, DC."
    answer = qa(question=question, context=context)

    print("=== QUESTION ===\n")
    print(question)
    print("\n=== CONTEXT ===\n")
    print(context)
    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()
