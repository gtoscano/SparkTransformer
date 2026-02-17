import argparse
import inspect

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def build_source_text(act_text: str) -> str:
    return (
        "Rewrite and improve this prompt so it is specific, actionable, and clear.\n"
        f"User goal: {act_text.strip()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a prompt improver model with Hugging Face datasets."
    )
    parser.add_argument("--dataset_id", default="fka/awesome-chatgpt-prompts")
    parser.add_argument("--model_id", default="t5-small")
    parser.add_argument("--output_dir", default="./prompt-improver-t5-small")
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_id, split="train")
    dataset = dataset.filter(
        lambda x: x.get("act") is not None and x.get("prompt") is not None
    )
    dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)

    def preprocess(batch):
        sources = [build_source_text(x) for x in batch["act"]]
        targets = [x.strip() for x in batch["prompt"]]

        model_inputs = tokenizer(
            sources,
            max_length=args.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )
    split = tokenized.train_test_split(test_size=0.1, seed=42)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        predict_with_generate=True,
        fp16=False,
        report_to="none",
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": split["train"],
        "eval_dataset": split["test"],
        "data_collator": DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    }
    trainer_init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to: {args.output_dir}")


if __name__ == "__main__":
    main()
