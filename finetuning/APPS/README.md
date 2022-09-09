# APPS finetuning
In this folder we show how to train an autoregressive Language model on APPS dataset, since a common way to evaluate on this benchmark is after finetuning the model on its training split.
We use Hugging Face [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) which supports distributed training on multiple GPUs.

## Setup

First login to Weights & Biases
```
wandb login
```

You can finetune a model, `gpt_345_python_any_license` for example, by running:
```python
# we use a global batch size of 256, here = 8 (GPUs) * 2 (batch_size_per_device) * 16 (gradient_accumulation)
python apps_train.py \
        --model_ckpt BigCode/gpt_345_python_any_license \
        --num_epochs 10 \
        --batch_size 2 \
        --gradient_accumulation_steps 16 \
        --learning_rate 5e-5 \
        --eval_freq 250 \
        --fp16
```
The fine-tuning takes 11h on 4 A100 GPUs.

## Acknowledgments

This script is adapted from [APPS repository](https://github.com/hendrycks/apps).