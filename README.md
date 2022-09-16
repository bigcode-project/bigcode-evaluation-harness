# Code Generation LM Evaluation Harness

A framework for the evaluation of autoregressive code generation language models. 

## Overview

This project provides a unified framework to test autoregressive code generation language models.

Features:
- Any autoregressive model available on [Hugging Face hub](https://huggingface.co/) can be used, but we recommend using a code generation models trained specifically on Code such as [CodeParrot](https://huggingface.co/codeparrot/codeparrot), [InCoder](https://huggingface.co/facebook/incoder-6B) and [CodeGen](https://huggingface.co/Salesforce/codegen-16B-mono).
- 3 code generation Python tasks implemented: [HumanEval](https://huggingface.co/datasets/openai_humaneval), [APPS](https://huggingface.co/datasets/codeparrot/apps) and [MBPP](https://huggingface.co/datasets/mbpp).
- [CoNaLa](https://huggingface.co/datasets/neulab/conala) for Python code generation (2-shot setting and evaluation with BLEU score)
- [Spider](https://huggingface.co/datasets/spider) for SQL code generation (2-shot setting and evaluation with BLEU score)
- Docstring generation task from code (zero-shot & fine-tuning) for 6 languages: Python, Go, Ruby, Java, JavaScript and PHP. 
- 3 multilingual downstream tasks: Java Complexity prediction, Java code equivalence prediction, C code defect prediction

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
# to run both humaneval and apps evaluations on Codeparrot with default parameters
accelerate launch main.py \
	--model codeparrot/codeparrot \
	--tasks humaneval,apps \
	--allow_code_execution=False
	
# to run full pass@k evaluation on the Introductory level of APPS (1000samples)
# the other levels are interview (3000 samples) and competitioon (1000 samples)
accelerate launch main.py \
	--model BigCode/gpt_all_license_apps_finetuned \
	--tasks apps \
	--level_apps introductory \
    	--n_samples 200 \
	--batch_size 40 \
	--temperature 0.6 \
	--allow_code_execution=False

# to evaluate only on some APPS samples using single generations
accelerate launch main.py \
	--model BigCode/gpt_all_license_apps_finetuned \
	--tasks apps \
	--level_apps introductory \
	--num_tasks_apps 10 \
    	--n_samples 1 \
	--batch_size 40 \
	--temperature 0.2 \
	--allow_code_execution=False
	
# to evaluate only on some MBPP samples with InCoder 1B
accelerate launch main.py \
	--model facebook/incoder-1B  \
	--prefix "<| file ext=.py |>\n" \
	--tasks mbpp \
	--num_tasks_mbpp 10 \
	--prompt_type_mbpp "incoder" \
    	--n_samples 1 \
	--temperature 0.2 \
	--allow_code_execution=False

# to evaluate on the code-to-text becnhmark
accelerate launch main.py \
	--model facebook/incoder-1B  \
	--prefix "<| file ext=.py |>\n" \
	--tasks conala \
	--num_tasks_conala 10 \
	--n_samples 1 
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


For APPS we use temperature 0.1:

   * Average accuracy:

<div align="center">
	
|Task | Model | Introductory| Interview| Competition| Average (weighted)|
|-------|--------|--------|-------|-------|-------|
|APPS | GPT2 finetuned (1.5B) | 10.01%| 7.52% | 4.29% | 7.37%|
|APPS | BigCode all license(340M) | **14.95%** | **10.47%**| **8.43%**| **10.95%**|
|APPS | BigCode safe license(340M) | 13.71% | 9.60% | 5.99%|9.7% |
	
</div>

* Strict accuracy:
<div align="center">

|Task | Model  | Introductory | Interview| Competition| Average (weighted)|
|-------|--------|--------|-------|-------|-------|
|APPS | GPT2 finetuned (1.5B) |2.4%| 0.5% | 0% |0.78% |
|APPS | BigCode all license(340M) | 3.4% | **0.7%**| **0.7%**|**1.24%**|
|APPS | BigCode safe license(340M) | **3.8%** |**0.7%** | 0.2%| 1.22%|
	
</div>

* Pass@k scores:

## Remarks
* Currenltly, we use parallel evaluation across multiple GPUs using `accelerate`, this assumes that you can fit the model in one GPU. 
* Please note this evaluation harness tries to cover a wide set of models, but there could still be room for improvement based on each model, some might require different prompt engineering or post-processing of the code generations.

## Evaluation time on 8 A100 GPUs:
- Evaluation on MBPP is 1-shot for 500 prompts, the evaluation takes **~1 hour**
- Evaluation on APPS (total of 5000 prompts) with single generations to compute average accuracy/strict accuracy takes in average **~4 hours for each of the 3 difficulty levels**
- The evaluation on APPS with multiple generations (nsamples=200) to compute pass@k takes **~16 hours for each 1000 samples (introductory level for example)**

## Acknowledgements
This repository is inspired from [EleutherAI's LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness).

## To do:
- [ ] add multilingual evaluation benchmarks
- [ ] test APPS one-shot setup
