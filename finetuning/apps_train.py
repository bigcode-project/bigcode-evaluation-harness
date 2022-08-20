"""
Fine-Tune LM on APPS train split
"""

import argparse
import os

from apps_dataset import APPSBaseDataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default="codeparrot/codeparrot-small")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    # parser.add_argument("--num_warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log-freq", default=1, type=int)
    parser.add_argument("--save-freq", default=50, type=int)
    return parser.parse_args()


def get_dataset(dataset, args):

    train_data = APPSBaseDataset(
        dataset=dataset, max_tokens=args.max_length, tokenizer_path=args.model_ckpt
    )

    return train_data


def run_training(args, train_data):

    model = AutoModelForCausalLM.from_pretrained(args.model_ckpt)
    train_data.start_iteration = 0

    print(f"Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        save_steps=args.save_freq,
        dataloader_drop_last=True,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        # warmup_steps = args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        run_name="apps-train",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    print("Training...")
    trainer.train()

    print("saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.save_dir, "final_checkpoint"))


def main(args):

    dataset = load_dataset("codeparrot/apps", split="train")
    dataset.shuffle(seed=args.seed)
    train_data = get_dataset(dataset, args)

    run_training(args, train_data)


if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
