# Code Generation LM Evaluation Harness

A framework for the evaluation of autoregressive code generation language models. 

## Overview

This project provides a unified framework to test autoregressive code generation language models.

Features:
- Any autoregressive model available on [Hugging Face hub](https://huggingface.co/) can be used, but we recommend using a code generation models trained specifically on Code such as [CodeParrot](https://huggingface.co/codeparrot/codeparrot), [InCoder](https://huggingface.co/facebook/incoder-6B) and [CodeGen](https://huggingface.co/Salesforce/codegen-16B-mono).
- 3 tasks implemented: [HumanEval](https://huggingface.co/datasets/openai_humaneval), [APPS](https://huggingface.co/datasets/codeparrot/apps) and [MBPP](https://huggingface.co/datasets/mbpp).


## Setup

```bash
git clone https://github.com/bigcode-collaboration/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
pip install -r requirements.txt
```
We used `accelerate` to generate code in parallel when multiple GPUs are present. You can configure it using:

```bash
accelerate config
```
## Basic Usage

Below are some examples to evaluate a model (CodeParrot and fine-tuned GPT2 on APPS) on HumanEval and APPS benchmarks:

```bash
#to run both humaneval and apps evaluations on Codeparrot with default parameters
accelerate launch main.py \
	--model codeparrot/codeparrot \
	--tasks humaneval,apps \
	--allow_code_execution=False

#to evaluate only on some APPS samples using single generations
accelerate launch main.py \
	--model BigCode/gpt_all_license_apps_finetuned \
	--tasks apps \
	--level_apps introductory \
	--num_tasks_apps 10 \
    	--n_samples 1 \
	--temperature 0.2 \
	--allow_code_execution=False

#to evaluate only on some MBPP samples with InCoder 1B
accelerate launch main.py \
	--model facebook/incoder-1B  \
	--prefix "<| file ext=.py |>\n" \
	--tasks mbpp \
	--num_tasks_mbpp 10 \
	--prompt_type_mbpp "incoder" \
    	--n_samples 1 \
	--temperature 0.2 \
	--allow_code_execution=False
```

## Scores

For all experiments with use the default setting of top-p sampling with `p=0.95` and adapt the temperature.

For MBPP we use temperature 0.1:

<div align="center">
	
|Task | Model  | pass@1 | 
|-------|--------|---------|
|MBPP | InCoder (1B) | 10.6% | 
|MBPP | CodeParrot (1.5B) | 0.2%(to be updated) |
|MBPP | BigCode-any-license (340M) | 17% |
|MBPP | BigCode-safe-license (340M) | 10.2% |
|MBPP | CodeGen-Mono (16B) | 42.4%(*) |
|MBPP | code-cushman-001/davinci-002 (Codex) | 45%/58%(*)(+) |	
</div>

(*) score reported on [CodeT](https://arxiv.org/pdf/2207.10397v1.pdf) paper

(+) these models are variants of Codex available thrigh OpenAI API, size is unknown


For APPS we use temperature 0.2:

   * Average accuracy:

<div align="center">
	
|Task | Model | Introductory| Interview| Competition| Average |
|-------|--------|--------|-------|-------|-------|
|APPS | GPT2 finetuned (1.5B) | 10.01%| 7.52% | 4.29% | 7.28%|
|APPS | CodeParrot (1.5B) |  | | | |
|APPS | BigCode (340M) |  | | | |
	
</div>

* Strict accuracy:
<div align="center">

|Task | Model  | Introductory | Interview| Competition| Average |
|-------|--------|--------|-------|-------|-------|
|APPS | GPT2 finetuned (1.5B) |2.4%| 0.5% | 0% |0.97% |
|APPS | CodeParrot (1.5B) |  | | | |
|APPS | BigCode (340M) |  | | | |
	
</div>

* Pass@k scores:

## Remarks
* Currenltly, we use parallel evaluation across multiple GPUs using `accelerate`, this assumes that you can fit the model in one GPU. 
* Please note this evaluation harness tries to cover a wide set of models, but there could still be room for improvement based on each model, some might require different prompt engineering or post-processing of the code generations.

## Evaluation time on 8 A100 GPUs:
- Evaluation on MBPP is 1-shot for 500 prompts, the evaluation takes **~1 hour**
- Evaluation on APPS (total of 5000 prompts) with single generations to compute average accuracy/strict accuracy takes in average **~4 hours for each of the 3 difficulty levels** (<ins> although</ins> average accuracy might not be very relevant and strict accuracy similar to pass@1 with one genertion usually very low)
- The evaluation on APPS with multiple generations (n=200) to compute pass@k takes ~**16 hours for each of the 3 difficulty level**

## Acknowledgements
This repository is inspired from [EleutherAI's LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness).

## To do:
- [ ] add APPS scores
