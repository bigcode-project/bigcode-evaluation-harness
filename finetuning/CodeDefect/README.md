# CodeDefect finetuning [WIP]
In this folder we show how to train an autoregressive on [CodeDefect](https://huggingface.co/datasets/code_x_glue_cc_defect_detection) dataset, for the problem of predicting if a code is insecure or not. We use Hugging Face [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) which supports distributed training on multiple GPUs.

## Setup

First login to Weights & Biases and to Hugging Face hub if you want to push your model to the hub:
```
wandb login
huggingface-cli login
```

To fine-tune a model on this dataset you can use the following command:

