# Code-to-text finetuning [WIP]
In this folder we show how to train an autoregressive on [Code-to-text](https://huggingface.co/datasets/code_x_glue_ct_code_to_text) dataset, for natural language comments generation from code. We use Hugging Face [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) which supports distributed training on multiple GPUs.

## Setup

First login to Weights & Biases and to Hugging Face hub if you want to push your model to the hub:
```
wandb login
huggingface-cli login
```

During the training, we use the code as input to the model and docstring as label. To fine-tune a model on the Python dataset for example, you can use the following command:
```python
python train.py \
    --model_ckpt codeparrot/codeparrot-small \
    --language Python \
    --num_epochs 30 \
    --batch_size 8 \
    --num_warmup_steps 10 \
    --learning_rate 5e-4 
    --push_to_hub True
```

For the 2-shot evaluation we use as a prompt
```
Generate comments for these code snippets:
Code:
$CODE1
Comment:
$DOCSTRING1

Code:
CODE2
Comment:
$DOCSTRING2

Code: $CODE
"""
```
