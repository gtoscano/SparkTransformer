from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from hf_auth import get_hf_token


BASE_MODEL = "google/pegasus-cnn_dailymail"
LOCAL_FINE_TUNED_DIR = "pegasus-samsum"


def build_zero_shot_pipe(token=None):
    return pipeline("summarization", model=BASE_MODEL, token=token)


def build_fine_tuned_pipe(token=None):
    checkpoint = Path(LOCAL_FINE_TUNED_DIR)
    if not checkpoint.exists():
        return None

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, token=token)
    return pipeline("summarization", model=model, tokenizer=tokenizer)


def main():
    token = get_hf_token(required=False)
    dataset = load_dataset("knkarthick/samsum", split="test[:1]")
    sample = dataset[0]

    dialogue = sample["dialogue"]
    reference = sample["summary"]

    zero_shot = build_zero_shot_pipe(token=token)
    zero_shot_summary = zero_shot(dialogue, max_length=80, min_length=20, do_sample=False)[0]["summary_text"]

    fine_tuned = build_fine_tuned_pipe(token=token)

    print("=== SAMSUM DIALOGUE ===\n")
    print(dialogue)

    print("\n=== REFERENCE SUMMARY ===\n")
    print(reference)

    print("\n=== ZERO-SHOT PEGASUS (CNN/DM) ===\n")
    print(zero_shot_summary)

    if fine_tuned is None:
        print("\n=== FINE-TUNED MODEL ===\n")
        print("No local 'pegasus-samsum' checkpoint found. Run 07_pegasus_samsum_finetune.py first if you want a fine-tuned comparison.")
    else:
        fine_tuned_summary = fine_tuned(dialogue, max_length=80, min_length=20, do_sample=False)[0]["summary_text"]
        print("\n=== FINE-TUNED PEGASUS ON SAMSUM ===\n")
        print(fine_tuned_summary)

    print("\nQuestion for students: How does domain mismatch change what the model keeps, drops, or misstates?")


if __name__ == "__main__":
    main()
