# CodeComplex finetuning
In this folder we show how to train an autoregressive on CodeComplex dataset, for algorithmic complexity prediction of Java programs. We use Hugging Face [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) which supports distributed training on multiple GPUs.

## Setup

First login to Weights & Biases and to Hugging Face hub if you want to push your model to the hub:
```
wandb login
huggingface-cli login
```

To fine-tune a model on this dataset, `microsoft/unixcoder-base-nine` for example, you can use the following command:

```python
python train.py \
    --model_ckpt microsoft/unixcoder-base-nine \
    --num_epochs 60 \
    --num_warmup_steps 10 \
    --batch_size 8 \
    --learning_rate 5e-4 
```
