from transformers import pipeline


MODEL_ID = "google/flan-t5-base"

PROMPTS = {
    "students": "Summarize the following text for undergraduate NLP students in 4 bullet points. Focus on the main ideas and keep the wording simple.\n\n{text}",
    "executive": "Write a short executive summary of the following text in 3 sentences. Focus on the most important takeaways only.\n\n{text}",
    "methods": "Summarize the following text in 4 bullet points, but include only methods, process details, or technical steps.\n\n{text}",
}


def load_text():
    with open("cua.txt", "r", encoding="utf-8") as handle:
        return handle.read().strip()


def main():
    text = load_text()[:2500]
    generator = pipeline("text2text-generation", model=MODEL_ID)

    for name, prompt_template in PROMPTS.items():
        prompt = prompt_template.format(text=text)
        output = generator(prompt, max_new_tokens=180, do_sample=False)[0]["generated_text"]

        print(f"\n{'=' * 80}\nCONTROL SETTING: {name}\n{'=' * 80}")
        print(output)

    print("\nQuestion for students: How does changing the audience or focus change the summary?")


if __name__ == "__main__":
    main()
