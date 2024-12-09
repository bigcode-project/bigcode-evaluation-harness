import argparse

from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ckpt", type=str, default="microsoft/unixcoder-base-nine"
    )
    parser.add_argument("--language", type=str, default="Python")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--freeze", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--model_hub_name", type=str, default="codeclone_model")
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    ds = load_dataset("code_x_glue_ct_code_to_text", args.language)

    print("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_ckpt, num_labels=2
    )
    model.config.pad_token_id = model.config.eos_token_id

    if args.freeze:
        for param in model.roberta.parameters():
            param.requires_grad = False

    def tokenize(example):
        if args.language == "Python":
            # remove docstring from code
            chunks = example["code"].split('"""')
            code = chunks[0].strip() + chunks[2]
        else:
            code = example["code"]
        inputs = tokenizer(
            code, padding="max_length", truncation=True, max_length=args.max_length
        )
        labels = tokenizer(
            example["docstring"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        ).input_ids
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "label": labels_with_ignore_index,
        }

    tokenized_datasets = ds.map(
        tokenize,
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        run_name=f"code-to-text-{args.language}",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    print("Training...")
    trainer.train()

    # push the model to the Hugging Face hub
    if args.push_to_hub:
        model.push_to_hub(args.model_hub_name)


if __name__ == "__main__":
    main()
