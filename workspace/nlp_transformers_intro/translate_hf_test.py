from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging


def main() -> None:
    hf_logging.set_verbosity_error()
    model_id = "Helsinki-NLP/opus-mt-en-es"  # Alternative: Helsinki-NLP/opus-mt-tc-big-en-es
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    examples = [
        "Hello everyone, welcome to our NLP class.",
        "Transformers are powerful models for language tasks.",
        "Today we are testing machine translation with Hugging Face.",
    ]

    print(f"Model: {model_id}\n")
    for text in examples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        output_ids = model.generate(**inputs, max_new_tokens=120)
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"EN: {text}")
        print(f"ES: {result}\n")


if __name__ == "__main__":
    main()
