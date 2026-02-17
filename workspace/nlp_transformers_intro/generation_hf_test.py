from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    prompts = [
        "In this NLP class, transformers are important because",
        "A simple explanation of attention is",
    ]

    print(f"Model: {model_id}\n")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )
        out = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Generated: {out}\n")


if __name__ == "__main__":
    main()
