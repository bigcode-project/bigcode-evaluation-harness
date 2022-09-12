# Code-to-text finetuning [WIP]
In this folder we show how to train an autoregressive on [Code-to-text](https://huggingface.co/datasets/https://huggingface.co/datasets/code_x_glue_ct_code_to_text) dataset, for natural language comments generation from code. We use Hugging Face [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) which supports distributed training on multiple GPUs.

## Setup

First login to Weights & Biases and to Hugging Face hub if you want to push your model to the hub:
```
wandb login
huggingface-cli login
```

For the 2-shot evaluation we use as a prompt
```
Generate comments for these code snippets:
Code:
CODE1
Comment:
"""DOCSTRING1"""
Code:
CODE2
Comment:
"""DOCSTRING2"""
Code: $CODE
"""
```

For fine-tuned models we train them on this 
```
input = $CODE
label = $DOCSTRING
```

To fine-tune a model on the Python dataset for example, you can use the following command:

