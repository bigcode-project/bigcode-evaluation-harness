# Finetuning

In this folder we show how to train an autoregressive Language model on APPS dataset, since a common way to evaluate on this benchmark is after finetuning the model on its training split.
We use Hugging Face [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) which supports distributed training on multiple GPUs.

You can finetune a model, codeparrot-small for example, by running:
```python
# we use a global batch size of 256, here = 4 (GPUs) * 8 (batch_size_per_device) * 8 (gradient_accumulation)
python apps_train.py \
        --model_ckpt codeparrot/codeparrot-small \
        --num_epochs 10 \
        --batch_size 8 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5
```