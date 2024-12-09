# CodeDefect finetuning
In this folder we show how to train an autoregressive on [CodeDefect](https://huggingface.co/datasets/code_x_glue_cc_defect_detection) dataset, for the problem of predicting if a code is insecure or not. We use Hugging Face [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) which supports distributed training on multiple GPUs.

## Setup

First login to Weights & Biases and to Hugging Face hub if you want to push your model to the hub:
```
wandb login
huggingface-cli login
```

To fine-tune a model on this dataset you can use the following command:
```python
python train.py \
    --model_ckpt microsoft/unixcoder-base-nine \
    --num_epochs 30 \
    --batch_size 8 \
    --num_warmup_steps 10 \
    --learning_rate 5e-4 
    --push_to_hub True
```
This will fine-tune your model, push it to the hub and print the evaluation accuracy on the test set.