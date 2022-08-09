# Code Generation LM Evaluation Harness [WIP]

A framework for the evaluation of autoregressive code generation language models. 

## Overview

This project provides a unified framework to test autoregressive code generation language models.

Features:
- Any autoregressive model available on [Hugging Face hub](https://huggingface.co/) can be used, but we recommend using a code generation models trained specifically on Code such as [CodeParrot](https://huggingface.co/codeparrot/codeparrot), [InCoder](https://huggingface.co/facebook/incoder-6B) and [CodeGen](https://huggingface.co/Salesforce/codegen-16B-mono).
- 3 tasks implemented: [HumanEval](https://huggingface.co/datasets/openai_humaneval), [APPS](https://huggingface.co/datasets/codeparrot/apps) and [MBPP](https://huggingface.co/datasets/mbpp).


## Setup

```bash
git clone https://github.com/loubnabnl/code-evaluation-harness.git
cd code-evaluation-harness
pip install -r requirements.txt
```
We used `accelerate` to generate code in parallel when multiple GPUs are present. You can configure it using:

```bash
accelerate config
```
## Basic Usage

Below are some examples to evaluate a model (CodeParrot and fine-tuned GPT2 on APPS) on HumanEval and APPS benchmarks:

```bash
#to run both evaluation on Codeparrot with default parameters
accelerate launch main.py \
	--model codeparrot/codeparrot \
	--tasks humaneval,apps \
	--allow_code_execution=False

#to evaluate only on some APPS samples 
accelerate launch main.py \
	--model loubnabnl/apps-1.5B-model  \
	--tasks apps \
	--level_apps introductory \
    --n_samples 1 \
	--allow_code_execution=False
```

## Acknowledgements
This repository is inspired from [EleutherAI's LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness).

## To do:
- [ ] finish APPS fix
- [ ] add MBPP benchmark
- [ ] add a table with CodeParrot evaluation scores
