from textwrap import wrap

from transformers import pipeline


MODEL_ID = "facebook/bart-large-cnn"
CHUNK_SIZE = 1800


def chunk_text(text, chunk_size):
    return wrap(text, chunk_size, break_long_words=False, replace_whitespace=False)


def load_text():
    with open("cua.txt", "r", encoding="utf-8") as handle:
        return handle.read().strip()


def main():
    text = load_text()
    chunks = chunk_text(text, CHUNK_SIZE)
    summarizer = pipeline("summarization", model=MODEL_ID)

    partial_summaries = []
    for idx, chunk in enumerate(chunks, start=1):
        summary = summarizer(chunk, max_length=90, min_length=30, do_sample=False)[0]["summary_text"]
        partial_summaries.append(summary)
        print(f"\n=== CHUNK {idx}/{len(chunks)} SUMMARY ===\n")
        print(summary)

    merged_input = " ".join(partial_summaries)
    final_summary = summarizer(merged_input, max_length=140, min_length=50, do_sample=False)[0]["summary_text"]

    print("\n" + "=" * 80)
    print("FINAL CHUNK-AND-MERGE SUMMARY")
    print("=" * 80 + "\n")
    print(final_summary)

    print("\nQuestion for students: What information might be lost between the chunk summaries and the final merged summary?")


if __name__ == "__main__":
    main()
